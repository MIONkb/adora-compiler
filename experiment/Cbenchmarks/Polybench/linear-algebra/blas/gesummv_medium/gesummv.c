/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gesummv.c: this file is part of PolyBench/C */

// #include <stdio.h>
// #include <unistd.h>
// #include <string.h>
// #include <math.h>

/* Include polybench common header. */
#include "rocket_polybench.h"

/* Include benchmark-specific header. */
#include "gesummv.h"
#include "adora_test.h"


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
// void gesummv(int n,
// 		    DATA_TYPE alpha,
// 		    DATA_TYPE beta,
// 		    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
// 		    DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
// 		    DATA_TYPE POLYBENCH_1D(tmp,N,n),
// 		    DATA_TYPE POLYBENCH_1D(x,N,n),
// 		    DATA_TYPE POLYBENCH_1D(y,N,n))
// {

void gesummv(
		    DATA_TYPE A[N][N] ,
		    DATA_TYPE B[N][N] ,
		    DATA_TYPE tmp[N] ,
		    DATA_TYPE x[N] ,
		    DATA_TYPE y[N] )
{
  int i, j;
  DATA_TYPE alpha = 1.5;
  DATA_TYPE beta = 1.2;

#pragma scop
  for (i = 0; i < N; i++)
    {
      tmp[i] = SCALAR_VAL(0.0);
      y[i] = SCALAR_VAL(0.0);
      for (j = 0; j < N; j++)
	  {
	    tmp[i] = A[i][j] * x[j] + tmp[i];
	    y[i] = B[i][j] * x[j] + y[i];
	  }
      y[i] = alpha * tmp[i] + beta * y[i];
    }
#pragma endscop

}