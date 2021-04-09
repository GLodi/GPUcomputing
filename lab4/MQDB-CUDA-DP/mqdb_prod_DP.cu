
#include "../../utils/MQDB/mqdb.h"
#include "../../utils/common.h"
#include "mqdb_ker.h"

#define BLOCK_SIZE 16     // block size

struct tms {
	double CPUtms;
	double GPUtmsNaive;
	double GPUtmsMQDB;
	double GPUtmsMQDBDynPar1;
	double GPUtmsMQDBDynPark;
	float density;
};


/*
 * Test on MQDB kernels
 */
void testKernelsMQDB(uint n, uint k, struct tms* times) {

	// mqdb host matrices
	mqdb A, B, C, C1;

	// mqdb device matrices
	mqdb d_A, d_B, d_C;

	// fill in
	A = mqdbConst(n, k, 10, 1);
	B = mqdbConst(n, k, 10, 1);
	C = mqdbConst(n, k, 10, 1);
	C1 = mqdbConst(n, k, 10, 1);

	ulong nBytes = n * n * sizeof(float);
	ulong kBytes = k * sizeof(uint);
	printf("Memory size required = %.1f (MB)\n",(float)nBytes/(1024.0*1024.0));

	// malloc and copy on device memory
	d_A.nBlocks = A.nBlocks;
	CHECK(cudaMalloc((void**)&d_A.blkSize, kBytes));
	CHECK(cudaMemcpy(d_A.blkSize, A.blkSize, kBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMalloc((void**)&d_A.elem, nBytes));
	CHECK(cudaMemcpy(d_A.elem, A.elem, nBytes, cudaMemcpyHostToDevice));
	d_B.nBlocks = B.nBlocks;
	CHECK(cudaMalloc((void**)&d_B.blkSize, kBytes));
	CHECK(cudaMemcpy(d_B.blkSize, B.blkSize, kBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMalloc((void**)&d_B.elem, nBytes));
	CHECK(cudaMemcpy(d_B.elem, B.elem, nBytes, cudaMemcpyHostToDevice));
	d_C.nBlocks = C.nBlocks;
	CHECK(cudaMalloc((void**)&d_C.blkSize, kBytes));
	CHECK(cudaMemcpy(d_C.blkSize, C.blkSize, kBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMalloc((void**)&d_C.elem, nBytes));
	CHECK(cudaMemset(d_C.elem, 0.0, nBytes));

	/***********************************************************/
	/*                    CPU MQDB product                     */
	/***********************************************************/
	printf("CPU MQDB product...\n");
	double start = seconds();
	mqdbProd(A,B,C);
	double CPUTime = seconds() - start;
	printf("   CPU elapsed time: %.5f (sec)\n\n", CPUTime);

	/***********************************************************/
	/*                     GPU mat product                     */
	/***********************************************************/
	printf("Kernel (naive) mat product...\n");
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
	start = seconds();
	matProd<<<grid, block>>>(d_A, d_B, d_C, n);
	CHECK(cudaDeviceSynchronize());
	double GPUtime1 = seconds() - start;
	printf("   elapsed time:                %.2f (sec)\n", GPUtime1);
	printf("   speedup vs CPU MQDB product: %.2f\n", CPUTime/GPUtime1);
	CHECK(cudaMemcpy(C1.elem, d_C.elem, nBytes, cudaMemcpyDeviceToHost));
	CHECK(cudaMemset(d_C.elem, 0.0, nBytes));
	checkResult(C,C1);
	//	mqdbDisplay(C1);

	/***********************************************************/
	/*                     GPU MQDB product                    */
	/***********************************************************/
	printf("Kernel MQDB product...\n");
	uint sdim = 0;
	start = seconds();
	for (uint i = 0; i < k; i++ ) {
		uint d = A.blkSize[i];
		mqdbBlockProd<<<grid, block>>>(d_A, d_B, d_C, sdim, d, n);
		sdim += d;
	}
	CHECK(cudaDeviceSynchronize());
	double GPUtime2 = seconds() - start;
	printf("   elapsed time:                    %.2f (sec)\n", GPUtime2);
	printf("   speedup vs CPU MQDB product:     %.2f\n", CPUTime/GPUtime2);
	printf("   speedup vs GPU std mat product:  %.2f\n", GPUtime1/GPUtime2);
	// copy the array 'C' back from the GPU to the CPU
	CHECK(cudaMemcpy(C1.elem, d_C.elem, nBytes, cudaMemcpyDeviceToHost));
	CHECK(cudaMemset(d_C.elem, 0.0, nBytes));
	checkResult(C,C1);

	/***********************************************************/
	/*              GPU MQDB dynamic par. GRID(1)              */
	/***********************************************************/
	start = seconds();
	printf("Kernel MQDB product with dynamic parall. GRID(1)...\n");
	mqdbProd<<< 1, 1 >>>(d_A, d_B, d_C, k, n);
	CHECK(cudaDeviceSynchronize());
	double GPUtime3 = seconds() - start;
	printf("   elapsed time:                        %.2f (sec)\n", GPUtime3);
	printf("   speedup vs CPU MQDB product:         %.2f\n", CPUTime/GPUtime3);
	printf("   speedup vs GPU std mat product:      %.2f\n", GPUtime1/GPUtime3);
	printf("   speedup vs GPU MQDB product:         %.2f\n", GPUtime2/GPUtime3);
	// copy the array 'C' back from the GPU to the CPU
	CHECK(cudaMemcpy(C1.elem, d_C.elem, nBytes, cudaMemcpyDeviceToHost));
	//	print_mqdb(C1);
	checkResult(C,C1);

	/***********************************************************/
	/*              GPU MQDB dynamic par. GRID(k)              */
	/***********************************************************/
	start = seconds();
	printf("Kernel MQDB product with dynamic parall. GRID(k)...\n");
	mqdbProdk<<< 1, k >>>(d_A, d_B, d_C, n);
	CHECK(cudaDeviceSynchronize());
	double GPUtime4 = seconds() - start;
	printf("   elapsed time:                        %.2f (sec)\n", GPUtime4);
	printf("   speedup vs CPU MQDB product:         %.2f\n", CPUTime/GPUtime4);
	printf("   speedup vs GPU std mat product:      %.2f\n", GPUtime1/GPUtime4);
	printf("   speedup vs GPU MQDB product:         %.2f\n", GPUtime2/GPUtime4);
	printf("   speedup vs GPU MQDB product GRID(1): %.2f\n", GPUtime3/GPUtime4);
	// copy the array 'C' back from the GPU to the CPU
	CHECK(cudaMemcpy(C1.elem, d_C.elem, nBytes, cudaMemcpyDeviceToHost));
	//	print_mqdb(C1);
	checkResult(C,C1);

	CHECK(cudaFree(d_A.elem));
	CHECK(cudaFree(d_B.elem));
	CHECK(cudaFree(d_C.elem));

	// collect times
	times->CPUtms = CPUTime;
	times->GPUtmsNaive = GPUtime1;
	times->GPUtmsMQDB = GPUtime2;
	times->GPUtmsMQDBDynPar1 = GPUtime3;
	times->GPUtmsMQDBDynPark = GPUtime4;
	float den = 0;
	for (uint j = 0; j < k; j++)
		den += A.blkSize[j]*A.blkSize[j];
	times->density = den/(n*n);
}

/*
 * main function
 */
int main(int argc, char *argv[]) {
	uint n = 24*1024;      // matrix size
	uint min_k = 10;       // max num of blocks
	uint max_k = 20;       // max num of blocks

	struct tms times[max_k-min_k+1];

	// multiple tests on kernels
	for (uint k = min_k; k <= max_k; k++) {
		printf("\n*****   k = %d --- (avg block size = %f)\n",k,(float)n/k);
		testKernelsMQDB(n, k, &times[k-min_k]);
	}

	FILE *fd;
	fd = fopen("res.csv", "w");
	if (fd == NULL) {
		perror("file error!\n");
		exit(1);
	}

	// write results on file
	fprintf(fd,"num blocks,");
		for (uint j = 0; j <= max_k-min_k; j++)
			fprintf(fd,"%d,",j+min_k);

	fprintf(fd,"\nCPU MQDB product,");
	for (uint j = 0; j <= max_k-min_k; j++)
		fprintf(fd,"%.4f,",times[j].CPUtms);

	fprintf(fd,"\nKernel mat product naive,");
	for (uint j = 0; j <= max_k-min_k; j++)
		fprintf(fd,"%.4f,",times[j].GPUtmsNaive);

	fprintf(fd,"\nKernel MQDB product,");
	for (uint j = 0; j <= max_k-min_k; j++)
		fprintf(fd,"%.4f,",times[j].GPUtmsMQDB);

	fprintf(fd,"\nKernel MQDB product with dynamic parall. GRID(1),");
	for (uint j = 0; j <= max_k-min_k; j++)
		fprintf(fd,"%.4f,",times[j].GPUtmsMQDBDynPar1);

	fprintf(fd,"\nKernel MQDB product with dynamic parall. GRID(k),");
	for (uint j = 0; j <= max_k-min_k; j++)
		fprintf(fd,"%.4f,",times[j].GPUtmsMQDBDynPark);

	fprintf(fd,"\ndensity,");
	for (uint j = 0; j <= max_k-min_k; j++)
		fprintf(fd,"%.4f,",times[j].density);

	fclose(fd);

	return 0;
}


