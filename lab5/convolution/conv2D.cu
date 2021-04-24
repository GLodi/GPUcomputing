#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../../utils/common.h"

#define DATA_WIDTH   (20*1024)
#define DATA_HEIGHT  (20*1024)
#define BLOCK_SIZE   8
#define MASK_RADIUS  2
#define MASK_SIZE    (2 * MASK_RADIUS + 1)
#define TILE_WIDTH   (BLOCK_SIZE + MASK_SIZE - 1)
#define DEBUG 0

// constant mem
__constant__ float M_dev[MASK_SIZE*MASK_SIZE];

/*
 * kernel for convolution 2D (it holds only if MASK_RADIUS < BLOCK_SIZE)
 */
__global__ void conv2D(float *A, float *B) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int RAD = MASK_RADIUS;
  int BmR = BLOCK_SIZE - RAD;
  int W = DATA_WIDTH;
  int H = DATA_HEIGHT;
	int m = MASK_SIZE;

	// shared mem
	__shared__ float A_s[TILE_WIDTH][TILE_WIDTH];

  // START SHARED MEMORY LOADING

  // 1. copy the tile upper halo 
  if ((threadIdx.y < RAD) ) {
    
    // left corner
    if (threadIdx.x < RAD && (x-RAD) >= 0 && (y-RAD) >= 0)
      A_s[threadIdx.y][threadIdx.x] = A[(y-RAD) * W + x - RAD];

    // right corner
    if (threadIdx.x >= BmR && (x+RAD) < W && (y-RAD) >= 0) 
      A_s[threadIdx.y][threadIdx.x + 2*RAD] = A[(y-RAD) * W + x + RAD];
    
    // edge
    if ((y-RAD) >= 0) 
      A_s[threadIdx.y][threadIdx.x + RAD] = A[(y-RAD) * W + x ];  
  }

  // 2. copy the tile bottom halo 
  if (threadIdx.y >= BmR) {
    
    // left corner
    if (threadIdx.x < RAD && (x-RAD) >= 0 && (y+RAD) < H)
      A_s[threadIdx.y + 2*RAD][threadIdx.x] = A[(y+RAD) * W + x - RAD];

    // right corner
    if (threadIdx.x >= BmR && (y+RAD) < H) 
      A_s[threadIdx.y + 2*RAD][threadIdx.x + 2*RAD] = A[(y+RAD) * W + x + RAD];
    
    // edge
    if ((y+RAD) < H) 
      A_s[threadIdx.y + 2*RAD][threadIdx.x + RAD] = A[(y+RAD) * W + x];  
  }

  // 3. copy the tile left-edge halo 
  if (threadIdx.x < RAD) 
    // edge
    if ((x-RAD) >= 0) 
      A_s[threadIdx.y + RAD][threadIdx.x] = A[y * W + x - RAD];  

  // 4. copy the tile right-edge halo 
  if (threadIdx.x >= BmR) 
    // edge
    if ((x+RAD) < W) 
      A_s[threadIdx.y + RAD][threadIdx.x + 2*RAD] = A[y * W + x + RAD];  
      

  // 5. copy the tile center <-> block
	A_s[RAD + threadIdx.y][RAD + threadIdx.x] = A[y*W+x];
	
  // END SHARED MEMORY LOADING

	__syncthreads();

	float conv_sum = 0.0;
	for (int i = 0; i < m; i++)
		for (int j = 0; j < m; j++)
			conv_sum += A_s[threadIdx.y+i][threadIdx.x+j] * M_dev[i*m + j];
	
  // store conv result
  B[x*W+y] = conv_sum;
}

/*
 * Average filter
 */
void Avg_mask(float *mask) {
	int n = MASK_SIZE;
	for (int i = 0; i < n*n; i++)
		mask[i] = (float) 1.0 / (n * n);
}


/*
 * main
 */
int main(void) {

  // check params
  if (MASK_RADIUS >= BLOCK_SIZE) {
    printf("ERROR: it holds only if MASK_RADIUS < BLOCK_SIZE!\n");
    return 1;
  }

	int nW = DATA_WIDTH;
  int nH = DATA_HEIGHT;
	int b = BLOCK_SIZE;

	float M[MASK_SIZE*MASK_SIZE]; // const size
	float *A, *B, *A_dev, *B_dev;
	int datasize = nW * nH * sizeof(float);
  int masksize = MASK_SIZE*MASK_SIZE * sizeof(float);

  printf("Data size: %.2f (MB)\n", (float)datasize/(1024.0*1024.0));
	printf("Initializing data...\n");
	A = (float *) malloc(datasize);
	B = (float *) malloc(datasize);

	// initialize data
	for (int i = 0; i < nH; i++)
		for (int j = 0; j < nW; j++)
			A[i*nW+j] = rand()%10;

  // initialize mask 
	Avg_mask(M);

#if DEBUG
	// print data
	printf("Print matrix A...\n");
	for (int i = 0; i < nH; i++) {
    if (i%8 == 0 && i>0)
      printf("\n");

		for (int j = 0; j < nW; j++)
      if (j%8 == 0 && j>0)
			  printf(" %0.0f ", A[i*nW+j]);
      else
        printf("%0.0f ", A[i*nW+j]);
		printf("\n");
	}

	printf("Print matrix M ...\n");
	for (int i = 0; i < MASK_SIZE; i++) {
		for (int j = 0; j < MASK_SIZE; j++)
			  printf(" %1.2f ", M[i * MASK_SIZE + j]);
		printf("\n");
	}
#endif

	// cuda allocation 
	CHECK(cudaMemcpyToSymbol(M_dev, M, masksize));
	CHECK(cudaMalloc((void **) &A_dev, datasize));
	CHECK(cudaMalloc((void **) &B_dev, datasize));
	CHECK(cudaMemcpy(A_dev, A, datasize, cudaMemcpyHostToDevice));
	
	// block, grid dims, kernel
	dim3 block(b, b);
	dim3 grid((nW+b-1)/b, (nH+b-1)/b);
  double iStart, iElaps;
	iStart = seconds();
	conv2D<<<grid, block>>>(A_dev, B_dev);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
	printf("\nconv2D<<<(%d,%d), (%d,%d)>>> elapsed time %f sec \n\n", grid.x, grid.y, block.x, block.y, iElaps);
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(B, B_dev, datasize, cudaMemcpyDeviceToHost));

#if DEBUG
	// print out data
	printf("Print results...\n");
	for (int i = 0; i < nH; i++) {
    if (i%8 == 0 && i>0)
      printf("\n");
		for (int j = 0; j < nW; j++)
      if (j%8 == 0 && j>0)
			  printf(" %0.2f ", B[i*nW+j]);
      else
        printf("%0.2f ", B[i*nW+j]);
		printf("\n");
	}
#endif

	cudaFree(A_dev);
	cudaFree(B_dev);
  cudaDeviceReset();
	free(A);
	free(B);
	return 0;
}

