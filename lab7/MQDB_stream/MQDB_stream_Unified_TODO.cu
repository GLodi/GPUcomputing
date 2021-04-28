

#include "../../utils/MQDB/mqdb.h"
#include "../../utils/common.h"

#define BLOCK_SIZE 16     // block size
#define TEST_CPU 0

/*
 * Kernel for standard (naive) matrix product
 */
__global__ void matProdKernel(mqdb *A, mqdb *B, mqdb *C, int n) {
	// row & col indexes
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread computes an entry of the product matrix
	if ((row < n) && (col < n)) {
		float val = 0;
		for (int k = 0; k < n; k++)
			val += A->elem[row * n + k] * B->elem[k * n + col];
		C->elem[row * n + col] = val;
	}
}

/*
 * Kernel for block sub-matrix product of mqdb
 */
__global__ void mqdbBlockProd(mqdb *A, mqdb *B, mqdb *C, uint sdim, uint d, uint n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// jump to the right block sub-matrix
	uint  offset = (n+1)*sdim;

	// each thread computes an entry of the product matrix
	if ((row < d) && (col < d)) {
		float val = 0;
		for (int k = 0; k < d; k++)
			val += A->elem[row * n + k + offset] * B->elem[k * n + col + offset];
		C->elem[row * n + col + offset] = val;
	}
}


/*
 * Test on MQDB kernels using Unified Memory
 */
void testKernelsMQDB_unified(uint n, uint k, cudaEvent_t start, cudaEvent_t stop) {

	// TODO
	
	/***********************************************************/
	/*                    CPU MQDB product                     */
	/***********************************************************/
	
  printf("CPU MQDB product...\n");
	double CPUtime = 0.0;

  #if TEST_CPU
    double startTm = seconds();
	  mqdbProd(A,B,C);
	  CPUtime = seconds() - startTm;
  #endif

	printf("   CPU elapsed time: %.5f (sec)\n\n", CPUtime);

	/***********************************************************/
	/*                     GPU mat product                     */
	/***********************************************************/
	
  printf("Kernel (naive) mat product...\n");

  // TODO - using matProdKernel


	/***********************************************************/
	/*                     GPU MQDB product                    */
	/***********************************************************/
	
  printf("Kernel MQDB product...\n");
	
  // TODO - using mqdbBlockProd

  /***********************************************************/
	/*             GPU MQDB product using streams              */
	/***********************************************************/
	
  printf("GPU MQDB product using streams...\n");

  // TODO - using mqdbBlockProd + streams

}

/*
 * main function
 */
int main(int argc, char *argv[]) {
  
  // set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s starting mqdb product at ", argv[0]);
	printf("device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// events to measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint n = 16*1024;         // matrix size
	uint min_k = 20;       // max num of blocks
	uint max_k = 30;       // max num of blocks

	// multiple tests for k = # diag blocks
	for (uint k = min_k; k <= max_k; k+=5) {
		printf("\n*****   k = %d --- (avg block size = %f)\n",k,(float)n/k);
		testKernelsMQDB_unified(n, k, start, stop);
	}

  cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}


