/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* bicg.c: this file is part of PolyBench/C */

// #include <stdio.h>
// #include <unistd.h>
// #include <string.h>
#include <math.h>

# define DATA_TYPE_IS_FLOAT

/* Include polybench common header. */
#include "rocket_polybench.h"

/* Include benchmark-specific header. */
#include "bicg.h"
#include "adora_test.h"

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
// void bicg(int m, int n,
// 		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
// 		 DATA_TYPE POLYBENCH_1D(s,M,m),
// 		 DATA_TYPE POLYBENCH_1D(q,N,n),
// 		 DATA_TYPE POLYBENCH_1D(p,M,m),
// 		 DATA_TYPE POLYBENCH_1D(r,N,n))
// {

void bicg(
		 DATA_TYPE A[N][M],//124 116
		 DATA_TYPE s[M],    //116
		 DATA_TYPE q[N],//124
		 DATA_TYPE p[M],//116
		 DATA_TYPE r[N] //124
     )
{
  int i, j;
  int m = M; // 116
  int n = N; // 124
#pragma scop
  for (i = 0; i < _PB_M; i++){
    s[i] = 0;
  }

  for (i = 0; i < _PB_N; i++)
    {
      q[i] = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_M; j++)
	    {
	      s[j] = s[j] + r[i] * A[i][j];
	      q[i] = q[i] + A[i][j] * p[j];
	    }
    }

  // for (i = 0; i < _PB_M; i++){
  //   s[i] = 0;
  //   q[i] = SCALAR_VAL(0.0);
  // }
  // for (j = 0; j < _PB_M; j++)
  //   {
  //     for (i = 0; i < _PB_N; i++)
	// {
	//   s[j] = s[j] + r[i] * A[i][j];
	//   q[i] = q[i] + A[i][j] * p[j];
	// }
  //   }

#pragma endscop

}
