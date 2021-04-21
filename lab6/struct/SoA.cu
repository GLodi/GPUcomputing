

#include <stdio.h>
#include <stdint.h>
#include "../../utils/common.h"

#define N 1<<24
#define blocksize 1<<7

struct SoA {
	uint8_t r[N];
	uint8_t g[N];
	uint8_t b[N];
};


void initialize(SoA*, int);
void checkResult(SoA*, SoA*, int);

/*
 * Riscala l'immagine al valore massimo [max] fissato
 */
__global__ void rescaleImg(SoA *img, const int max, const int n) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		float r,g,b;
		SoA *tmp = img;
		r = max * (float)tmp->r[i]/255.0f;
		img->r[i] = (uint8_t)r;
		g = max * (float)tmp->g[i]/255.0f;
		img->g[i] = (uint8_t)g;
		b = max * (float)tmp->b[i]/255.0f;
		img->b[i] = (uint8_t)b;
	}
}

/*
 * cancella un piano dell'immagine [plane = 'r' o 'g' o 'b'] fissato
 */
__global__ void deletePlane(SoA *img, const char plane, const int n) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		switch (plane) {
		case 'r':
			img->r[i] = 0;
			break;
		case 'g':
			img->g[i] = 0;
			break;
		case 'b':
			img->b[i] = 0;
			break;
		}
	}
}

/*
 * setup device
 */
__global__ void warmup(SoA *img, const int max, const int n) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		float r,g,b;
		SoA *tmp = img;
		r = max * (float)tmp->r[i]/255.0f;
		img->r[i] = (uint8_t)r;
		g = max * (float)tmp->g[i]/255.0f;
		img->g[i] = (uint8_t)g;
		b = max * (float)tmp->b[i]/255.0f;
		img->b[i] = (uint8_t)b;
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
	printf("%s test SoA at ", argv[0]);
	printf("device %d: %s \n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// scelta del kernel da eseguire
	int kernel = 0;
	if (argc > 1) kernel = atoi(argv[1]);

	// allocate host memory
	size_t nBytes = sizeof(SoA);
	SoA *img = (SoA *)malloc(nBytes);
	SoA *new_img = (SoA *)malloc(nBytes);

	// initialize host array
	initialize(img, N);

	// allocate device memory
	int n_elem = N;
	SoA *d_img;
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
	if (kernel == 0)
		rescaleImg<<<grid, block>>>(d_img, max, n_elem);
	else
		deletePlane<<<grid, block>>>(d_img, 'r', n_elem);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	printf("rescaleImg <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x, iElaps);
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

void initialize(SoA *img,  int size) {
	for (int i = 0; i < size; i++) {
		img->r[i] = rand() % 256;
		img->g[i] = rand() % 256;
		img->b[i] = rand() % 256;
	}
	return;
}

void checkResult(SoA *img, SoA *new_img, int n_elem) {
	for (int i = 0; i < n_elem; i+=1000)
		printf("img[%d] = (%d,%d,%d) -- new_img[%d] = (%d,%d,%d)\n",
				i,img->r[i],img->g[i],img->b[i],i,new_img->r[i],new_img->g[i],new_img->b[i]);
	return;
}


void transposeHost(float *out, float *in, const int nx, const int ny) {
	for (int iy = 0; iy < ny; ++iy) {
		for (int ix = 0; ix < nx; ++ix) {
			out[ix * ny + iy] = in[iy * nx + ix];
		}
	}
}

