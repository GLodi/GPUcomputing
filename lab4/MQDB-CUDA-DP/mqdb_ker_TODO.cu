
#include "../../utils/MQDB/mqdb.h"
#include "mqdb_ker.h"

#define BLOCK_SIZE 16     // block size

/*
 * Kernel for standard (naive) matrix product
 */
__global__ void matProd(mqdb A, mqdb B, mqdb C, uint n) {
	// row & col indexes
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread computes an entry of the product matrix
	if ((row < n) && (col < n)) {
		float val = 0;
		for (int k = 0; k < n; k++)
			val += A.elem[row * n + k] * B.elem[k * n + col];
		C.elem[row * n + col] = val;
	}
}

/*
 * Kernel for block sub-matrix product of mqdb
 */
__global__ void mqdbBlockProd(mqdb A, mqdb B, mqdb C, uint sdim, uint d, uint n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// jump to the right block sub-matrix
	int  offset = (n+1)*sdim;

	// each thread computes an entry of the product matrix
	if ((row < d) && (col < d)) {
		float val = 0;

		for (int k = 0; k < d; k++)
			val += A.elem[row * n + k + offset] * B.elem[k * n + col + offset];
		C.elem[row * n + col + offset] = val;
	}
}

/*
 * Kernel for block sub-matrix product of mqdb: parent grid(1)
 */
__global__ void mqdbProd(mqdb A, mqdb B, mqdb C, uint k, uint n) {
	// using grid(1,1)
	
  // TODO
}

/*
 * Kernel for block sub-matrix product of mqdb: parent grid(k)
 */
__global__ void mqdbProdk(mqdb A, mqdb B, mqdb C, uint n) {
	// using grid(1,k)
	
  // TODO
  
}
