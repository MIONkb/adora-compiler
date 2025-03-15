/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* atax.c: this file is part of PolyBench/C */

// #include <stdio.h>
// #include <unistd.h>
// #include <string.h>
// #include <math.h>

#include <math.h>

# define DATA_TYPE_IS_FLOAT

/* Include polybench common header. */
#include "rocket_polybench.h"

/* Include benchmark-specific header. */
#include "atax.h"
#include "adora_test.h"

// void atax(int m, int n,
// 		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
// 		 DATA_TYPE POLYBENCH_1D(x,N,n),
// 		 DATA_TYPE POLYBENCH_1D(y,N,n),
// 		 DATA_TYPE POLYBENCH_1D(tmp,M,m))


void atax(
		 DATA_TYPE A[M][N],
		 DATA_TYPE x[N],
		 DATA_TYPE y[N],
		 DATA_TYPE tmp[M])
{
  int i, j;

#pragma scop
  for (i = 0; i < N; i++)
    y[i] = 0;
  for (i = 0; i < M; i++)
    {
      tmp[i] = SCALAR_VAL(0.0);
      for (j = 0; j < N; j++)
	      tmp[i] = tmp[i] + A[i][j] * x[j];
      for (j = 0; j < M; j++)
	      y[j] = y[j] + A[i][j] * tmp[i];
    }
#pragma endscop

}
