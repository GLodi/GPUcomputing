/*
 * matrix.h
 *
 * Matrices are stored in row-major order:
 * M(i,j) corresponds to *(M.elem + i * M.cols + j)
 */

#ifndef MQDB_H
#define MQDB_H

typedef struct MQDB {
	char desc[100];  // description
	int num_blks;    // num. of blocks
	int *blk_dims;   // block dimensions
	float* elems;    // elements in row-major order
} mqdb;

// function prototype
mqdb rand_gen_mqdb(unsigned, unsigned, unsigned);
mqdb const_mqdb(unsigned, unsigned, unsigned, float);
void mat_prod_H(mqdb, mqdb, mqdb);
void mqdb_prod_H(mqdb, mqdb, mqdb);
void checkResult(mqdb, mqdb);
void print_mqdb(mqdb);

#endif
