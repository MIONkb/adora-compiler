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

/* Array initialization. */
static
void init_array (int m,
		 int n,
		 DATA_TYPE *float_n,
    //DATA_TYPE POLYBENCH_2D(data,N,M,n,m)
		 DATA_TYPE data [N][M]
     )
{
  int i, j;

  *float_n = (DATA_TYPE)N;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = (DATA_TYPE)(i*j)/M + i;

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		//  DATA_TYPE POLYBENCH_2D(corr,M,M,m,m)
    DATA_TYPE corr[M][M]
     )

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("corr");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      // if ((i * m + j) % 20 == 0) printf (POLYBENCH_DUMP_TARGET, "\n");
      // fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, corr[i][j]);
      if ((i * m + j) % 20 == 0) printf ("\n");
      printf (DATA_PRINTF_MODIFIER, corr[i][j]);
    }
  POLYBENCH_DUMP_END("corr");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
// void kernel_correlation(int m, int n,
// 			DATA_TYPE float_n,
// 			DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
// 			DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
// 			DATA_TYPE POLYBENCH_1D(mean,M,m),
// 			DATA_TYPE POLYBENCH_1D(stddev,M,m))
// {
void kernel_correlation(
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


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  // POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  // POLYBENCH_2D_ARRAY_DECL(corr,DATA_TYPE,M,M,m,m);
  // POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  // POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);

  DATA_TYPE data[N][M];
  DATA_TYPE corr[M][M];
  DATA_TYPE mean[M];
  DATA_TYPE stddev[M];

  /* Initialize array(s). */
  init_array (M, N, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  // kernel_correlation (m, n, float_n,
	// 	      POLYBENCH_ARRAY(data),
	// 	      POLYBENCH_ARRAY(corr),
	// 	      POLYBENCH_ARRAY(mean),
	// 	      POLYBENCH_ARRAY(stddev));

  kernel_correlation ( float_n,
		      data,
		      corr,
		      mean,
		      stddev);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  // polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(corr)));
  print_array(m, POLYBENCH_ARRAY(corr));

  // /* Be clean. */
  // POLYBENCH_FREE_ARRAY(data);
  // POLYBENCH_FREE_ARRAY(corr);
  // POLYBENCH_FREE_ARRAY(mean);
  // POLYBENCH_FREE_ARRAY(stddev);

  return 0;
}


