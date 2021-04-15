
#include <stdlib.h>
#include <stdio.h>
#include "../../utils/common.h"

#define MASK_RADIUS  5
#define MASK_SIZE    2 * MASK_RADIUS + 1
#define TILE         128
#define TILE_SIZE    TILE + MASK_SIZE - 1

__device__ __constant__ float d_mask[MASK_SIZE];

void initialData(float*, int);
void movingAverage(float*, int n);
void printData(float*, const int);
void convolutionHost(float*, float*, float*, const int);
void checkResult(float*, float*, int);

/*
 * kernel: 1D convolution
 */
__global__ void convolution1D(float *result, float *data, int n) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	// shared memory size = TILE + MASK
	__shared__ float tile[TILE_SIZE];

	// boundary
	int left = blockIdx.x * blockDim.x - MASK_RADIUS;
	int right = (blockIdx.x + 1) * blockDim.x;

  // left halo
	if (threadIdx.x < MASK_RADIUS)                      
		tile[threadIdx.x] = left < 0 ? 0 : data[left + threadIdx.x];

  // center
	tile[threadIdx.x + MASK_RADIUS] = data[i];

  // right halo  
	if (threadIdx.x >= blockDim.x - MASK_RADIUS)  
		tile[threadIdx.x + MASK_SIZE - 1] = right >= n ? 0 :
				data[right + threadIdx.x - blockDim.x + MASK_RADIUS];

	__syncthreads();

	// convolution: tile * mask
	float sum = 0;
	for (int i = -MASK_RADIUS; i <= MASK_RADIUS; i++)
		sum += tile[threadIdx.x + MASK_RADIUS + i] * d_mask[i + MASK_RADIUS];

	// final result
	result[i] = sum;
}

/*
 * MAIN: convolution 1D host & device
 */
int main(int argc, char **argv) {
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("starting conv1D at device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// set up array size
	int n = 1 << 24;
	int N = MASK_SIZE;

	printf("Array of size = %.1f MB\n", n/(1024.0*1024.0));

	// mem sizes
	size_t nBytes = n * sizeof(float);
	size_t nBytes_mask = N * sizeof(float);

	// grid configuration
	dim3 block(TILE);
	dim3 grid((n + TILE - 1) / TILE);

	// allocate host memory
	float *h_data = (float *) malloc(nBytes);
	float *h_result = (float *) malloc(nBytes);
	float *result = (float *) malloc(nBytes);
	float *h_mask = (float *) malloc(nBytes_mask);

	//  initialize host array
	movingAverage(h_mask, N);
	initialData(h_data, n);

	// convolution on host
	double start = seconds();
	convolutionHost(h_data, result, h_mask, n);
	double hostElaps = seconds() - start;

	// allocate device memory
	float *d_data, *d_result;
	CHECK(cudaMalloc((void**)&d_data, nBytes));
	CHECK(cudaMalloc((void**)&d_result, nBytes));

	// copy data from host to device
	CHECK(cudaMemcpy(d_data, h_data, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_mask, h_mask, nBytes_mask));

	start = seconds();
	convolution1D<<<grid, block>>>(d_result, d_data, n);
	CHECK(cudaDeviceSynchronize());
	double devElaps = seconds() - start;
  printf("Times:\n");
	printf("   - CPU elapsed time = %f\n", hostElaps);
  printf("   - GPU elapsed time = %f\n", devElaps);
  printf("   - Speed-up (ratio) = %f\n", hostElaps / devElaps);

	CHECK(cudaMemcpy(h_result, d_result, nBytes, cudaMemcpyDeviceToHost));

	// check result
	checkResult(h_result, result, n);

	// free host and device memory
	CHECK(cudaFree(d_result));
	CHECK(cudaFree(d_data));
	free(h_data);
	free(h_mask);
	free(h_result);
	free(result);

	// reset device
	CHECK(cudaDeviceReset());
	return EXIT_SUCCESS;
}

void initialData(float *h_data, int n) {
	// initialize the data
	for (int i = 0; i < n; i++)
		h_data[i] = 10.0;
}

void movingAverage(float *h_mask, int n) {
	// initialize mask moving average
	for (int i = 0; i < n; i++)
		h_mask[i] = 1.0 / ((float) n);
	return;
}

void printData(float *a, const int size) {
	printf("\n");
	for (int i = 0; i < size; i++)
		printf("%.2f ", a[i]);
	printf("\n");
	return;
}

void convolutionHost(float *data, float *result, float *mask, const int n) {
	for (int i = 0; i < n; i++) {
		float sum = 0;
		for (int j = 0; j < MASK_SIZE; j++) {
			int idx = i - MASK_RADIUS + j;
			if (idx >= 0 && idx < n)
				sum += data[idx] * mask[j];
		}
		result[i] = sum;
	}
}

void checkResult(float *d_result, float *h_result, int n) {
	double epsilon = 1.0E-8;

	for (int i = 0; i < n; i++)
		if (abs(h_result[i] - d_result[i]) > epsilon) {
			printf("different on entry (%d) |h_result - d_result| >  %f\n", i,
					epsilon);
			break;
		}
}

