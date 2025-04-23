/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* correlation.c: this file is part of PolyBench/C */

// #include <stdio.h>
// #include <unistd.h>
// #include <string.h>
#include <math.h>

/* Include polybench common header. */
// #include <polybench.h>
#include "rocket_polybench.h"

/* Include benchmark-specific header. */
#include "correlation.h"
#include "cgrv_test.h"

#ifndef ROCKET_TARGET
#define ROCKET_TARGET
#endif


# define POLYBENCH_USE_SCALAR_LB
# define POLYBENCH_USE_C99_PROTO
# define DATA_TYPE_IS_FLOAT
#ifdef ROCKET_TARGET
  #define DATA_TYPE_IS_FLOAT
// [[gnu::noinline]]
// void kernel_correlation(int m, int n,
// 			DATA_TYPE float_n,
// 			DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
// 			DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
// 			DATA_TYPE POLYBENCH_1D(mean,M,m),
// 			DATA_TYPE POLYBENCH_1D(stddev,M,m)) __attribute__((noinline));
[[gnu::noinline]]
void kernel_correlation(
			DATA_TYPE float_n,
			DATA_TYPE data[N][M],
			DATA_TYPE corr[M][M],
			DATA_TYPE mean[M],
			DATA_TYPE stddev[M]) __attribute__((noinline));

#else
void kernel_correlation(int m, int n,
			DATA_TYPE float_n,
			DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
			DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
			DATA_TYPE POLYBENCH_1D(mean,M,m),
			DATA_TYPE POLYBENCH_1D(stddev,M,m));
#endif


#define DATA_TYPE float
// #define size_t int32_t



/* Main computational kernel. The whole function will be timed,
   including the call and return. */
// void kernel_correlation(int m, int n,
// 			DATA_TYPE float_n,
// 			DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
// 			DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
// 			DATA_TYPE POLYBENCH_1D(mean,M,m),
// 			DATA_TYPE POLYBENCH_1D(stddev,M,m))
// {
void correlation(
			DATA_TYPE float_n,
			DATA_TYPE data[N][M],
			DATA_TYPE corr[M][M],
			DATA_TYPE mean[M],
			DATA_TYPE stddev[M])
{
  int n = N, m = M;
  int i, j, k;
  float_n = (DATA_TYPE)N;
  DATA_TYPE eps = SCALAR_VAL(0.1);


#pragma scop
//////////////// Start of Kernel0
  // //origin kernel1
  // for (j = 0; j < _PB_M; j++)
  //   {
  //     mean[j] = SCALAR_VAL(0.0);
  //     for (i = 0; i < _PB_N; i++)
	//       mean[j] += data[i][j];
  //     mean[j] /= float_n;
  //   }
  for(j = 0; j < _PB_M; j++){
     mean[j] = SCALAR_VAL(0.0);
  }

  for(i = 0; i<_PB_N;i++){
    for(j = 0; j < _PB_M; j++){
      mean[j] += data[i][j];
    }
  }

  for(j = 0; j < _PB_M; j++){
    mean[j] /= float_n;
  }
  //////////////// End of Kernel0

  //////////////// Start of Kernel1
  for (j = 0; j < _PB_M; j++)
  {
    stddev[j] = SCALAR_VAL(0.0);
  }


  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);

  // origin kernel1
  for (j = 0; j < _PB_M; j++)
  {
    stddev[j] = SCALAR_VAL(0.0);
    for (i = 0; i < _PB_N; i++)
      stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
    stddev[j] /= float_n;
    stddev[j] = SQRT_FUN(stddev[j]);
    /* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
    stddev[j] = stddev[j] <= eps ? SCALAR_VAL(1.0) : stddev[j];
  }

  //////////////// End of Kernel1

  /* Center and reduce the column vectors. */
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= SQRT_FUN(float_n) * stddev[j];
      }

  /* Calculate the m * m correlation matrix. */
  for (i = 0; i < _PB_M-1; i++)
    {
      corr[i][i] = SCALAR_VAL(1.0);
      for (j = i+1; j < _PB_M; j++)
        {
          corr[i][j] = SCALAR_VAL(0.0);
          for (k = 0; k < _PB_N; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[_PB_M-1][_PB_M-1] = SCALAR_VAL(1.0);
#pragma endscop

}