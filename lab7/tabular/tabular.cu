
#include <stdio.h>
#include "../../utils/common.h"

#define PI 3.141592f

/*
 * Kernel: tabular function
 */
__global__ void tabular(float *a, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		float x = PI * (float)i / (float)n;
		float s = sinf(x);
		float c = cosf(x);
		a[i] = sqrtf(abs(s * s - c * c));
	}
}

/*
 * Kernel: tabular function using streams
 */
__global__ void tabular_streams(float *a, int n, int offset) {
	int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float x = PI * (float)i / (float)n;
    float s = sinf(x);
    float c = cosf(x);
    a[i] = sqrtf(abs(s * s - c * c));
  }
}

/*
 * Error measure
 */
float maxError(float *a, int n) {
	float maxE = 0;
	for (int i = 0; i < n; i++) {
		float error = fabs(a[i] - 1.0f);
		if (error > maxE)
			maxE = error;
	}
	return maxE;
}

/*
 * Main: tabular function
 */
int main(void) {
	
  // main params
  uint MB = 1024*1024; 
  uint n = 256*MB;
	int blockSize = 256;
	int nStreams = 8;

	int streamSize = n / nStreams;
	int streamBytes = streamSize * sizeof(float);
	int bytes = n * sizeof(float);

	int devId = 0;
	cudaDeviceProp prop;
	CHECK(cudaGetDeviceProperties(&prop, devId));
	printf("Device : %s\n\n", prop.name);
	CHECK(cudaSetDevice(devId));
  printf("Array size   : %d\n", n);
  printf("StreamSize   : %d\n", streamSize);
  printf("Memory bytes : %d (MB)\n", bytes/MB);
  printf("streamBytes  : %d (MB)\n", streamBytes/MB);

	// allocate pinned host memory and device memory
	float *a, *d_a;
	CHECK(cudaMallocHost((void**) &a, bytes));      // host pinned
	CHECK(cudaMalloc((void**) &d_a, bytes));        // device

	float ms; // elapsed time in milliseconds

	// create events and streams
	cudaEvent_t startEvent, stopEvent, dummyEvent;
	cudaStream_t stream[nStreams];
	CHECK(cudaEventCreate(&startEvent));
	CHECK(cudaEventCreate(&stopEvent));
	CHECK(cudaEventCreate(&dummyEvent));
	for (int i = 0; i < nStreams; ++i)
		CHECK(cudaStreamCreate(&stream[i]));

	// baseline case - sequential transfer and execute
	memset(a, 0, bytes);
	CHECK(cudaEventRecord(startEvent, 0));
	CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
	tabular<<<n / blockSize, blockSize>>>(d_a, n);
	CHECK(cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost));
	CHECK(cudaEventRecord(stopEvent, 0));
	CHECK(cudaEventSynchronize(stopEvent));
	CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	printf("\nTime for sequential transfer and execute (ms): %f\n", ms);
	printf("  max error: %e\n", maxError(a, n));

	// asynchronous version 1: loop over {copy, kernel, copy}
	memset(a, 0, bytes);
	CHECK(cudaEventRecord(startEvent, 0));
	for (int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;
		CHECK(cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes,cudaMemcpyHostToDevice, stream[i]));
		tabular_streams<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a,n,offset);
		CHECK(cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]));
	}
	CHECK(cudaEventRecord(stopEvent, 0));
	CHECK(cudaEventSynchronize(stopEvent));
	CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	printf("\nTime for asynchronous V1 transfer and execute (ms): %f\n", ms);
	printf("  max error: %e\n", maxError(a, n));

	// asynchronous version 2:
	// loop over copy, loop over kernel, loop over copy
	memset(a, 0, bytes);
	CHECK(cudaEventRecord(startEvent, 0));
	for (int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;
		CHECK(cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes,cudaMemcpyHostToDevice, stream[i]));
	}
	for (int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;
		tabular_streams<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a,n,offset);
	}
	for (int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;
		CHECK(cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes,cudaMemcpyDeviceToHost, stream[i]));
	}
	CHECK(cudaEventRecord(stopEvent, 0));
	CHECK(cudaEventSynchronize(stopEvent));
	CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	printf("\nTime for asynchronous V2 transfer and execute (ms): %f\n", ms);
	printf("  max error: %e\n", maxError(a, n));

	// cleanup
	CHECK(cudaEventDestroy(startEvent));
	CHECK(cudaEventDestroy(stopEvent));
	CHECK(cudaEventDestroy(dummyEvent));
	for (int i = 0; i < nStreams; ++i)
		CHECK(cudaStreamDestroy(stream[i]));
	cudaFree(d_a);
	cudaFreeHost(a);

	return 0;
}
