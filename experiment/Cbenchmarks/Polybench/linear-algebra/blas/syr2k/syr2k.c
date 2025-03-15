/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* syr2k.c: this file is part of PolyBench/C */

// #include <stdio.h>
// #include <unistd.h>
// #include <string.h>
// #include <math.h>

#include "rocket_polybench.h"

/* Include benchmark-specific header. */
#include "adora_test.h"

/* Include benchmark-specific header. */
// #define MINI_DATASET
// #define SMALL_DATASET
// #define MEDIUM_DATASET
#define LARGE_DATASET

/* Include benchmark-specific header. */
#include "syr2k.h"

// void syr2k(int n, int m,
// 		  DATA_TYPE alpha,
// 		  DATA_TYPE beta,
// 		  DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
// 		  DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
// 		  DATA_TYPE POLYBENCH_2D(B,N,M,n,m))

void syr2k(
		  DATA_TYPE alpha,
		  DATA_TYPE beta,
		  DATA_TYPE C[N][N],
		  DATA_TYPE A[N][M],
		  DATA_TYPE B[N][M])
{
  int i, j, k;

//BLAS PARAMS
//UPLO  = 'L'
//TRANS = 'N'
//A is NxM
//B is NxM
//C is NxN
#pragma scop
  for (i = 0; i < N; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < M; k++)
      for (j = 0; j <= i; j++)
	{
	  C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k];
	}
  }
#pragma endscop

}
