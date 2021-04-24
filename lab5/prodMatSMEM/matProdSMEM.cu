
#include <stdio.h>
#include <stdlib.h>
#include "../../utils/common.h"

#define IDX(i,j,n) (i*n+j)
#define ABS(x,y) (x-y>=0?x-y:y-x)
#define N 1024
#define P 1024
#define M 1024

#define BLOCK_SIZE 16

/*
 * Kernel for matrix product with static SMEM
 *      C  =  A  *  B
 *    (NxM) (MxP) (PxM)
 */
__global__ void matProdSMEMstatic(float* A, float* B, float* C) {
	// indexes
	uint row = blockIdx.y * blockDim.y + threadIdx.y;
	uint col = blockIdx.x * blockDim.x + threadIdx.x;

	// target: compute the right sum for the given row and col
	float sum = 0.0;

	// static shared memory
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	// loop over blocks from block row of matrix A
	// and block column of matrix B
	uint numBlocks = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
	for (uint m = 0; m < numBlocks; m++) {

		// copy block from matrix to shared memory
		uint r = m * BLOCK_SIZE + threadIdx.y;
		uint c = m * BLOCK_SIZE + threadIdx.x;
		As[threadIdx.y][threadIdx.x] = A[IDX(row, c, P)];
		Bs[threadIdx.y][threadIdx.x] = B[IDX(r, col, M)];

		//---------------------------------------------------------------
		__syncthreads();  //  BARRIER SYNC on SMEM loading

		// length of this part of row-column product is BLOCK_SIZE
		// except for last block when it may be smaller
		uint K = BLOCK_SIZE;
		if (m == numBlocks - 1) K = P - m * BLOCK_SIZE; // tune last block

		// compute this part of row-column product
		for (uint k = 0; k < K; k++)
			sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

		//---------------------------------------------------------------
		__syncthreads();  //  BARRIER SYNC on prod over blocks
		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
	}

	// store computed element in matrix C
	if (row < N && col < M)
		C[row * M + col] = sum;
}


/*
 * Kernel for matrix product using dynamic SMEM
 */
__global__ void matProdSMEMdynamic(float* A, float* B, float* C, const uint SMEMsize) {
	// indexes
	uint row = blockIdx.y * blockDim.y + threadIdx.y;
	uint col = blockIdx.x * blockDim.x + threadIdx.x;

	// dynamic shared memory (inside or outside kernel)
	extern __shared__ float smem[];

	// Var As is manually set at beginning of shared
	float *As = smem;
	// Var Bs is manually set at the end of As
	float *Bs = &smem[SMEMsize];

	// loop over blocks from block row of matrix A
	// and block column of matrix B
	float sum = 0.0;
	uint numBlocks = (P + blockDim.x - 1) / blockDim.x;
	for (uint m = 0; m < numBlocks; m++) {

		// copy block from matrix to shared memory
		uint c = m * blockDim.x + threadIdx.x;
		uint r = m * blockDim.y + threadIdx.y;
		As[threadIdx.y * blockDim.y + threadIdx.x] = A[IDX(row, c, P)];
		Bs[threadIdx.y * blockDim.y + threadIdx.x] = B[IDX(r, col, M)];

		//---------------------------------------------------------------
		__syncthreads();

		// length of this part of row-column product is BLOCK_SIZE
		// except for last block when it may be smaller
		uint K = (m == numBlocks - 1 ? P - m * blockDim.x : blockDim.x);

		// compute this part of row-column product
		for (int k = 0; k < K; k++)
			sum += As[threadIdx.y * blockDim.x + k] * Bs[k * blockDim.y + threadIdx.x];

		//---------------------------------------------------------------
		__syncthreads();
	}

	// store computed element in matrix C
	if (row < N && col < M)
		C[row * M + col] = sum;
}

/*
 * Kernel for naive matrix product
 */
__global__ void matProd(float* A, float* B, float* C) {
	// indexes
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread computes an entry of the product matrix
	if ((row < N) && (col < M)) {
		float sum = 0;
		for (int k = 0; k < P; k++)
			sum += A[row * P + k] * B[k * M + col];
		C[row * M + col] = sum;
	}
}

/*
 *  matrix product on CPU
 */
void matProdCPU(float* A, float* B, float* C) {

	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++) {
			float sum = 0;
			for (int k = 0; k < P; k++)
				sum += A[i * P + k] * B[k * M + j];
			C[i * M + j] = sum;
		}
}

/*
 * Test the device
 */
unsigned long testCUDADevice(void) {
	int dev = 0;

	cudaDeviceSetCacheConfig (cudaFuncCachePreferEqual);
	cudaDeviceProp deviceProp;
	cudaSetDevice(dev);
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Device %d: \"%s\"\n", dev, deviceProp.name);
	printf("Total amount of shared memory available per block: %lu KB\n",
			deviceProp.sharedMemPerBlock / 1024);
	return deviceProp.sharedMemPerBlock;
}


/*
 * elementwise comparison between two mqdb
 */
void checkResult(float *A, float *B) {
	double epsilon = 1.0E-8;
	bool match = 1;
	for (int i = 0; i < N*M; i++)
		if (ABS(A[i], B[i]) > epsilon) {
			match = 0;
			printf("   * Arrays do not match!\n");
			break;
		}
	if (match)
		printf("   Arrays match\n\n");
}

/*
 * MAIN
 */
int main(void) {
	 // Kernels for matrix product
	 //      C  =  A  *  B
	 //    (NxM) (MxP) (PxM)
	uint rowA = N, rowB = P;
	uint colA = P, colB = M;
	uint rowC = N, colC = M;
	float *A, *B, *C, *C1;
	float *dev_A, *dev_B, *dev_C;

	// dims
	unsigned long Asize = rowA * colA * sizeof(float);
	unsigned long Bsize = rowB * colB * sizeof(float);
	unsigned long Csize = rowC * colC * sizeof(float);
	unsigned long maxSMEMbytes;
	uint nByteSMEM = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
	printf("N = %d, M = %d, P = %d\n",N,M,P);

	// test device shared memory
	maxSMEMbytes = testCUDADevice();
	if (maxSMEMbytes < nByteSMEM)
		printf("Shared memory usage WARNING: available: %lu, required: %d bytes\n",
				maxSMEMbytes, nByteSMEM);
	else
		printf("Total amount of shared memory required per block %.1f KB\n",
				(float) nByteSMEM / (float) 1024);

	// malloc host memory
	A = (float*) malloc(Asize);
	B = (float*) malloc(Bsize);
	C = (float*) malloc(Csize);
	C1 = (float*) malloc(Csize);

	// malloc device memory
	CHECK(cudaMalloc((void** )&dev_A, Asize));
	CHECK(cudaMalloc((void** )&dev_B, Bsize));
	CHECK(cudaMalloc((void** )&dev_C, Csize));
	printf("Total amount of allocated memory on GPU %lu bytes\n\n",
			Asize + Bsize + Csize);

	// fill the matrices A and B
	for (int i = 0; i < N * P; i++)
		A[i] = rand() % 10;
	for (int i = 0; i < P * M; i++)
		B[i] = rand() % 10;
	matProdCPU(A, B, C);

	// copy matrices A and B to the GPU
	CHECK(cudaMemcpy(dev_A, A, Asize, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_B, B, Bsize, cudaMemcpyHostToDevice));

	/***********************************************************/
	/*              GPU matProdSMEM static SMEM               */
	/***********************************************************/
	// grid block dims = shared mem dims = BLOCK_SIZE
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
	double start = seconds();
	matProdSMEMstatic<<<grid, block>>>(dev_A, dev_B, dev_C);
	CHECK(cudaDeviceSynchronize());
	printf("   Kernel matProdSMEM static elapsed time GPU = %f\n", seconds() - start);

	// copy the array 'C' back from the GPU to the CPU
	CHECK(cudaMemcpy(C1, dev_C, Csize, cudaMemcpyDeviceToHost));
	checkResult(C,C1);

	/***********************************************************/
	/*            GPU matProdSMEMD dynamic SMEM                */
	/***********************************************************/
	// set cache size
	cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);

	// try with various SMEM sizes
	uint sizes[] = {8,16,32};
	for (int i = 0; i < 3; i++) {
		uint blockSize = sizes[i];
		block.x = blockSize;
		block.y = blockSize;
		grid.x = (M + block.x - 1) / block.x;
		grid.y = (N + block.y - 1) / block.y;
		uint SMEMsize = blockSize * blockSize;
		uint SMEMbyte = 2 * SMEMsize * sizeof(float);
		start = seconds();
		matProdSMEMdynamic<<< grid, block, SMEMbyte >>>(dev_A, dev_B, dev_C, SMEMsize);
		CHECK(cudaDeviceSynchronize());
		printf("   Kernel matProdSMEM dynamic (SMEM size %d) elapsed time GPU = %f\n", blockSize, seconds() - start);

		// copy the array 'C' back from the GPU to the CPU
		CHECK(cudaMemcpy(C1, dev_C, Csize, cudaMemcpyDeviceToHost));
		checkResult(C,C1);
	}

	// free the memory allocated on the GPU
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);

	cudaDeviceReset();
	return EXIT_SUCCESS;
}
