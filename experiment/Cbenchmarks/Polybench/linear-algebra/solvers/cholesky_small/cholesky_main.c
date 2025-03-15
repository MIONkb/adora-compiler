/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* cholesky.c: this file is part of PolyBench/C */

// #include <stdio.h>
// #include <unistd.h>
// #include <string.h>
#include <math.h>

/* Include polybench common header. */
// #include <polybench.h>
#include "rocket_polybench.h"

/* Include benchmark-specific header. */
#ifndef ROCKET_TARGET
#define ROCKET_TARGET
#define __riscv
#define __GNUC__
#define SMALL_DATASET
#endif
# define SMALL_DATASET
# define POLYBENCH_USE_SCALAR_LB
# define POLYBENCH_USE_C99_PROTO
# define DATA_TYPE_IS_FLOAT
#include "cholesky.h"

#include "cgrv_test.h"
#include "include/encoding.h"


/* Array initialization. */
static
void init_array(int n,
		 DATA_TYPE A[N][N])
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite. */
  int r,s,t;
  // POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  DATA_TYPE B[N][N];
  // for (r = 0; r < n; ++r)
  //   for (s = 0; s < n; ++s)
  //     DATA_TYPE B[r][s] = 0;
  //     // (POLYBENCH_ARRAY(B))[r][s] = 0;
  // for (t = 0; t < n; ++t)
  //   for (r = 0; r < n; ++r)
  //     for (s = 0; s < n; ++s)
	// (POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
  //   for (r = 0; r < n; ++r)
  //     for (s = 0; s < n; ++s)
	// A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
  // POLYBENCH_FREE_ARRAY(B);

    for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	      B[r][s] += A[r][t] * A[s][t];
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	      A[r][s] = B[r][s];

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
// static
// void print_array(int n,
// 		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
static
void print_array(int n,
		 DATA_TYPE A[N][N])

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j <= i; j++) {
    // if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    // fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    if ((i * n + j) % 20 == 0) printf ("\n");
    printf (DATA_PRINTF_MODIFIER, A[i][j]);
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
// static
// void kernel_cholesky(int n,
// 		     DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
// {
//   int i, j, k;


// #pragma scop
//   for (i = 0; i < _PB_N; i++) {
//      //j<i
//      for (j = 0; j < i; j++) {
//         for (k = 0; k < j; k++) {
//            A[i][j] -= A[i][k] * A[j][k];
//         }
//         A[i][j] /= A[j][j];
//      }
//      // i==j case
//      for (k = 0; k < i; k++) {
//         A[i][i] -= A[i][k] * A[i][k];
//      }
//      A[i][i] = SQRT_FUN(A[i][i]);
//   }
// #pragma endscop

// }

void cholesky(DATA_TYPE A[N][N]);

int main(int argc, char** argv)
{
  printf("start cholesky noprint with init!\n");
  printf("N: %d!\n", N);
  /* Retrieve problem size. */
  int n = N;

  long long unsigned start;
  long long unsigned end;

  /* Variable declaration/allocation. */
  // POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  DATA_TYPE A[N][N];

  /* Initialize array(s). */
  printf("Initialization\n");
  init_array (n, A);
  printf("Initialization finished!\n");

  // /* Start timer. */
  // polybench_start_instruments;

  /* Run kernel. */
  printf("start kernel\n");
  start = rdcycle();
  cholesky (A);
  end = rdcycle();

  printf("It takes %llu cycles for CPU to finish the task.\n", end - start);
  printf("start: %llu\n", start);
  printf("end: %llu\n", end);

  // /* Stop and print timer. */
  // polybench_stop_instruments;
  // polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  // polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));
  printf("start printing\n");
  print_array(n, A);

  /* Be clean. */
  // POLYBENCH_FREE_ARRAY(A);

  return 0;
}
