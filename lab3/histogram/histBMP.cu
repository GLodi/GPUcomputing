/**
 * hist.cu
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>

#include "bmpUtil.h"
#include "common.h"

/*
 * Kernel 1D that computes histogram on GPU
 */
__global__ void histogramBMP(uint *bins, const pel *imgSrc, const uint W, const uint N, const uint M) {
	// ** pixel granularity **
	uint x = blockDim.x * blockIdx.x + threadIdx.x; // 1D pixel linear index over [0:W*H)
	uint nrows = x / W;                             // num of rows to skip
	uint off = x % W;                               // offset (= col) within current row

	if (x >= N)                        // pixel out of range
		return;

	//  ** byte granularity **
	uint p = M * nrows + 3*off;        // src byte position of the pixel
	pel R = imgSrc[p];
	pel G = imgSrc[p+1];
	pel B = imgSrc[p+2];
	atomicAdd(&bins[R], 1);
	atomicAdd(&bins[G+256], 1);
	atomicAdd(&bins[B+512], 1);
}

/*
 * Function that computes histogram on CPU
 */
void hist_CPU(uint *bins, const pel *imgSrc, const uint W, const uint H, const uint M) {
	for (int i = 0; i < W*H; i++) {
		uint r = i / W;              // row of the source pixel
		uint off = i - r * W;        // col of the source pixel

		//  ** byte granularity **
		uint p = M * r + 3*off;      // src byte position of the pixel
		pel R = imgSrc[p];
		pel G = imgSrc[p+1];
		pel B = imgSrc[p+2];
		bins[R] += 1;
		bins[G+256] += 1;
		bins[B+512] += 1;
	}
}

int main(int argc, char **argv) {

	uint dimBlock = 1024;
	pel *imgBMP_CPU;     // Where images are stored in CPU
	pel *imgBMP_GPU;	 // Where images are stored in GPU

	uint *binsRGB_CPU, *binsRGB_GPU, *binsRGB_GPU2CPU;
	uint N_bins = 3*256;
	uint bin_size = N_bins*sizeof(uint);

	if (argc > 2)
		dimBlock = atoi(argv[2]);
	else if (argc < 2) {
		printf("\n\nUsage:  hist InputFilename dimBlock\n");
		exit(EXIT_FAILURE);
	}

	// bins for CPU & GPU
	binsRGB_CPU = (uint*) calloc(N_bins, sizeof(uint));
	binsRGB_GPU2CPU = (uint*) malloc(bin_size);
	CHECK(cudaMalloc((void**) &binsRGB_GPU, bin_size));

	// Create CPU memory to store the input image
	imgBMP_CPU = ReadBMPlin(argv[1]);
	if (imgBMP_CPU == NULL) {
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}

	// Allocate GPU buffer for image and bins
	CHECK(cudaMalloc((void**) &imgBMP_GPU, IMAGESIZE));

	// Copy input vectors from host memory to GPU buffers.
	CHECK(cudaMemcpy(imgBMP_GPU, imgBMP_CPU, IMAGESIZE, cudaMemcpyHostToDevice));

	// CPU histogram
	double start = seconds();   // start time
	hist_CPU(binsRGB_CPU, imgBMP_CPU, WIDTH, HEIGHT, WIDTHB);
	double stop = seconds();   // elapsed time
	printf("\nCPU elapsed time %f sec \n\n", stop - start);

	// invoke kernels (define grid and block sizes)
	uint nPixels = WIDTH*HEIGHT;
	int dimGrid = (nPixels + dimBlock - 1) / dimBlock;
	printf("\ndimGrid = %d   dimBlock = %d\n",dimGrid,dimBlock);

	start = seconds();   // start time
	histogramBMP<<<dimGrid, dimBlock>>>(binsRGB_GPU, imgBMP_GPU, WIDTH, nPixels, WIDTHB);
	CHECK(cudaDeviceSynchronize());
	stop = seconds();   // elapsed time
	printf("\nGPU elapsed time %f sec \n\n", stop - start);

	// Copy output (results) from GPU buffer to host (CPU) memory.
	CHECK(cudaMemcpy(binsRGB_GPU2CPU, binsRGB_GPU, bin_size, cudaMemcpyDeviceToHost));

	for (int i = 0; i < N_bins/3; i++)
		printf("bin_GPU[%d] = \t%d\t%d\t%d\t -- bin_CPU[%d] = \t%d\t%d\t%d\n", i,
				binsRGB_GPU2CPU[i],binsRGB_GPU2CPU[i+256],binsRGB_GPU2CPU[i+512],
				i,binsRGB_CPU[i],binsRGB_CPU[i+256],binsRGB_CPU[i+512]);

	// Deallocate GPU memory
	cudaFree(imgBMP_GPU);
	cudaFree(binsRGB_GPU);

	// tracing tools spel as Parallel Nsight and Visual Profiler to show complete traces.
	CHECK(cudaDeviceReset());

	return (EXIT_SUCCESS);
}

/*
 *  Read a 24-bit/pixel BMP file into a 1D linear array.
 *  Allocate memory to store the 1D image and return its pointer
 */
pel *ReadBMPlin(char* fn) {
	static pel *Img;
	FILE* f = fopen(fn, "rb");
	if (f == NULL) {
		printf("\n\n%s NOT FOUND\n\n", fn);
		exit(EXIT_FAILURE);
	}

	pel HeaderInfo[54];
	size_t nByte = fread(HeaderInfo, sizeof(pel), 54, f); // read the 54-byte header
	// extract image height and width from header
	int width = *(int*) &HeaderInfo[18];
	img.width = width;
	int height = *(int*) &HeaderInfo[22];
	img.height = height;
	int RowBytes = (width * 3 + 3) & (~3);  // row is multiple of 4 pixel
	img.rowByte = RowBytes;
	//save header for re-use
	memcpy(img.headInfo, HeaderInfo, 54);
	printf("\n Input File name: %5s  (%d x %d)   File Size=%lu", fn, img.width, img.height, IMAGESIZE);

	// allocate memory to store the main image (1 Dimensional array)
	Img = (pel *) malloc(IMAGESIZE);
	if (Img == NULL)
		return Img;      // Cannot allocate memory
	// read the image from disk
	size_t out = fread(Img, sizeof(pel), IMAGESIZE, f);
	fclose(f);
	return Img;
}
