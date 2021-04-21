

#include <stdio.h>
#include <stdint.h>
#include "../../utils/common.h"

#define N 1<<24
#define blocksize 128

struct AoS {
	uint8_t r;
	uint8_t g;
	uint8_t b;
};

void initialize(AoS *, int);
void checkResult(AoS *, AoS *, int);

/*
 * Riscala l'immagine al valore massimo [max] fissato
 */
__global__ void rescaleImg(AoS *img, const int max, const int n) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		float r,g,b;
		AoS tmp = img[i];
		r = max * (float)tmp.r/255.0f;
		tmp.r = (uint8_t)r;
		g = max * (float)tmp.g/255.0f;
		tmp.g = (uint8_t)g;
		b = max * (float)tmp.b/255.0f;
		tmp.b = (uint8_t)b;
		img[i] = tmp;
	}
}

/*
 * cancella un piano dell'immagine [plane = 'r' o 'g' o 'b'] fissato
 */
__global__ void deletePlane(AoS *img, const char plane, const int n) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		switch (plane) {
		case 'r':
			img[i].r = 0;
			break;
		case 'g':
			img[i].g = 0;
			break;
		case 'b':
			img[i].b = 0;
			break;
		}
	}
}

/*
 * setup device
 */
__global__ void warmup(AoS *img, const int max, const int n) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		float r,g,b;
		AoS tmp = img[i];
		r = max * (float)tmp.r/255.0f;
		tmp.r = (uint8_t)r;
		g = max * (float)tmp.g/255.0f;
		tmp.g = (uint8_t)g;
		b = max * (float)tmp.b/255.0f;
		tmp.b = (uint8_t)b;
		img[i] = tmp;
	}
}

/*
 * Legge da stdin quale kernel eseguire: 0 per rescaleImg, 1 per deletePlane
 */
int main(int argc, char **argv) {
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s test AoS at ", argv[0]);
	printf("device %d: %s \n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// scelta del kernel da eseguire
	int kernel = 0;
	if (argc > 1) kernel = atoi(argv[1]);

	// allocate host memory
	int n_elem = N;
	size_t nBytes = n_elem * sizeof(AoS);
	AoS *img = (AoS *)malloc(nBytes);
	AoS *new_img = (AoS *)malloc(nBytes);

	// initialize host array
	initialize(img, N);

	// allocate device memory
	AoS *d_img;
	CHECK(cudaMalloc((void**)&d_img, nBytes));

	// copy data from host to device
	CHECK(cudaMemcpy(d_img, img, nBytes, cudaMemcpyHostToDevice));

	// definizione max
	int max = 128;
	if (argc > 2) max = atoi(argv[2]);

	// configurazione per esecuzione
	dim3 block (blocksize, 1);
	dim3 grid  ((n_elem + block.x - 1) / block.x, 1);

	// kernel 1: warmup
	double iStart = seconds();
	warmup<<<1, 32>>>(d_img, max, 32);
	CHECK(cudaDeviceSynchronize());
	double iElaps = seconds() - iStart;
	printf("warmup<<< 1, 32 >>> elapsed %f sec\n",iElaps);
	CHECK(cudaGetLastError());

	// kernel 2 rescaleImg o deletePlane
	iStart = seconds();
	if (kernel == 0) {
		rescaleImg<<<grid, block>>>(d_img, max, n_elem);
		printf("rescaleImg <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x, iElaps);
	}
	else {
		deletePlane<<<grid, block>>>(d_img, 'r', n_elem);
		printf("deletePlane <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x, iElaps);
	}
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	CHECK(cudaMemcpy(new_img, d_img, nBytes, cudaMemcpyDeviceToHost));
	CHECK(cudaGetLastError());

	//checkResult(img, new_img, n_elem);

	// free memories both host and device
	CHECK(cudaFree(d_img));
	free(img);
	free(new_img);

	// reset device
	CHECK(cudaDeviceReset());
	return EXIT_SUCCESS;
}

void initialize(AoS *img,  int size) {
	for (int i = 0; i < size; i++) {
		img[i].r = rand() % 256;
		img[i].g = rand() % 256;
		img[i].b = rand() % 256;
	}
	return;
}

void checkResult(AoS *img, AoS *new_img, int n_elem) {
	for (int i = 0; i < n_elem; i+=1000)
		printf("img[%d] = (%d,%d,%d) -- new_img[%d] = (%d,%d,%d)\n",
				i,img[i].r,img[i].g,img[i].b,i,new_img[i].r,new_img[i].g,new_img[i].b);
	return;
}

