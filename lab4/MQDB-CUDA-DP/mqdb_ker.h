
#ifndef _MQDB_KER_H
#define _MQDB_KER_H

__global__ void matProd(mqdb, mqdb, mqdb, uint);
__global__ void mqdbBlockProd(mqdb, mqdb, mqdb, uint, uint, uint);
__global__ void mqdbProd(mqdb, mqdb, mqdb, uint, uint);
__global__ void mqdbProdk(mqdb, mqdb, mqdb, uint);

#endif 
