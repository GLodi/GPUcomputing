#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../../utils/common.h"

/*
 * Kernel with warp divergence
 */
__global__ void evenOddDIV(int *c, const ulong N) {
	ulong tid = blockIdx.x * blockDim.x + threadIdx.x;
	int a, b;

	if (!(tid % 2))   // branch divergence
		a = 2;                  
	else
		b = 1;                  

	// check index
	if (tid < N)
		c[tid] = a + b;
}

/*
 * Kernel without warp divergence
 */
__global__ void evenOddNODIV(int *c, const int N) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int a = 0, b = 0;
	unsigned int i, twoWarpSize = 2 * warpSize;

	int wid = tid / warpSize; 	// warp index wid = 0,1,2,3,...
	if (!(wid % 2))
		a = 2;                  // branch1: thread tid = 0-31, 64-95, ...
	else
		b = 1;                  // branch2: thread tid = 32-63, 96-127, ...

	// right index
	if (!(wid % 2))  // even
		i = 2 * (tid % warpSize) + (tid / twoWarpSize) * twoWarpSize;
	else            // odd
		i = 2 * (tid % warpSize) + 1 + (tid / twoWarpSize) * twoWarpSize;

	// check index
	if (i < N) {
		c[i] = a + b;
	}
}

/*
 * MAIN
 */
int main(int argc, char **argv) {

	// set up data size
	int blocksize = 1024;
	ulong size = 1024*1024;

	if (argc > 1)
		blocksize = atoi(argv[1]);
	if (argc > 2)
		size = atoi(argv[2]);
	ulong nBytes = size * sizeof(int);

	printf("Data size: %lu  -- ", size);
  printf("Data size (bytes): %lu MB\n", nBytes/1000000);

	// set up execution configuration
	dim3 block(blocksize, 1);
	dim3 grid((size + block.x - 1) / block.x, 1);
	printf("Execution conf (block %d, grid %d)\nKernels:\n", block.x, grid.x);

	// allocate memory
	int *d_C, *C;
	C = (int *) malloc(nBytes);
	CHECK(cudaMalloc((void** )&d_C, nBytes));

	// run kernel 1
	double iStart, iElaps;
	iStart = seconds();
	evenOddDIV<<<grid, block>>>(d_C, size);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	printf("\tevenOddDIV<<<%d, %d>>> elapsed time %f sec \n\n", grid.x, block.x, iElaps);
	CHECK(cudaGetLastError());
  
  CHECK(cudaMemcpy(C, d_C, nBytes, cudaMemcpyDeviceToHost));


	// run kernel 2
  CHECK(cudaMemset(d_C, 0.0, nBytes)); // reset memory
	iStart = seconds();
	evenOddNODIV<<<grid, block>>>(d_C, size);
	iElaps = seconds() - iStart;
	printf("\tevenOddNODIV<<<%d, %d>>> elapsed time %f sec \n\n", grid.x, block.x, iElaps);
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(C, d_C, nBytes, cudaMemcpyDeviceToHost));

	free(C);
	// free gpu memory and reset device
	CHECK(cudaFree(d_C));
	CHECK(cudaDeviceReset());
	return EXIT_SUCCESS;
}
