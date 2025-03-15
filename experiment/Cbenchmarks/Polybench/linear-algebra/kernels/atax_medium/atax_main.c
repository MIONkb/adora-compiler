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

/* Include polybench common header. */
#include "rocket_polybench.h"

/* Include benchmark-specific header. */
#ifndef ROCKET_TARGET
#define ROCKET_TARGET
#define __riscv
#define __GNUC__
#define MEDIUM_DATASET
#endif
# define MEDIUM_DATASET
# define POLYBENCH_USE_SCALAR_LB
# define POLYBENCH_USE_C99_PROTO
# define DATA_TYPE_IS_FLOAT
// #include <polybench.h>
// # define DATA_TYPE float
/* Include benchmark-specific header. */
#include "atax.h"
#include "adora_test.h"
#include "include/encoding.h"

/* Array initialization. */
// static
// void init_array (int m, int n,
// 		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
// 		 DATA_TYPE POLYBENCH_1D(x,N,n))
// {

static
void init_array (
		 DATA_TYPE A[M][N],
		 DATA_TYPE x[N])
{
  int i, j;
  int n = N;
  int m = M;
  DATA_TYPE fn;
  fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
      x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % n) / (5*m);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE y[N])

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) printf ("\n");
    // fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
    printf (DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
// static
// void kernel_atax(int m, int n,
// 		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
// 		 DATA_TYPE POLYBENCH_1D(x,N,n),
// 		 DATA_TYPE POLYBENCH_1D(y,N,n),
// 		 DATA_TYPE POLYBENCH_1D(tmp,M,m))
// {
//   int i, j;

// #pragma scop
//   for (i = 0; i < _PB_N; i++)
//     y[i] = 0;
//   for (i = 0; i < _PB_M; i++)
//     {
//       tmp[i] = SCALAR_VAL(0.0);
//       for (j = 0; j < _PB_N; j++)
// 	      tmp[i] = tmp[i] + A[i][j] * x[j];
//       for (j = 0; j < _PB_N; j++)
// 	      y[j] = y[j] + A[i][j] * tmp[i];
//     }
// #pragma endscop

// }



void atax(
		 DATA_TYPE A[M][N],
		 DATA_TYPE x[N],
		 DATA_TYPE y[N],
		 DATA_TYPE tmp[M]);


int main(int argc, char** argv)
{
  printf("CGRA start atax medium without init!\n");
  printf("loaddata is simplified\n");
  printf("M: %d, N: %d\n", M, N);

  /* Retrieve problem size. */
  int m = M;
  int n = N;

  long long unsigned start;
  long long unsigned end;

  /* Variable declaration/allocation. */
  // POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  // POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  // POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  // POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, M, m);
	DATA_TYPE A[M][N];
	DATA_TYPE x[N];
	DATA_TYPE y[N];
	DATA_TYPE tmp[M];


  /* Initialize array(s). */
  // init_array (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));
  // init_array(A, x);

  /* Start timer. */
  // polybench_start_instruments;

  /* Run kernel. */
  // kernel_atax (m, n,
	//        POLYBENCH_ARRAY(A),
	//        POLYBENCH_ARRAY(x),
	//        POLYBENCH_ARRAY(y),
	//        POLYBENCH_ARRAY(tmp));

  printf("\nstart kernel\n");
  start = rdcycle();
  atax(A, x, y, tmp);
  end = rdcycle();

  printf("It takes %llu cycles for CPU to finish the task.\n", end - start);
  printf("start: %llu\n", start);
  printf("end: %llu\n", end);

  /* Stop and print timer. */
  // polybench_stop_instruments;
  // polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  // polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));
  printf("start printing\n"); 
  print_array(n, y);
  /* Be clean. */
  // POLYBENCH_FREE_ARRAY(A);
  // POLYBENCH_FREE_ARRAY(x);
  // POLYBENCH_FREE_ARRAY(y);
  // POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}
