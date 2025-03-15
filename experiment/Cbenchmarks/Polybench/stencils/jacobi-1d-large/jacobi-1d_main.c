/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-1d.c: this file is part of PolyBench/C */

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
#define LARGE_DATASET
#endif
# define LARGE_DATASET
# define POLYBENCH_USE_SCALAR_LB
# define POLYBENCH_USE_C99_PROTO
# define DATA_TYPE_IS_FLOAT
#include "jacobi-1d.h"

#include "adora_test.h"
#include "include/encoding.h"

/* Array initialization. */
// static
// void init_array (int n,
// 		 DATA_TYPE POLYBENCH_1D(A,N,n),
// 		 DATA_TYPE POLYBENCH_1D(B,N,n))

static
void init_array (int n,
		 DATA_TYPE A[N],
		 DATA_TYPE B[N])
{
  int i;

  for (i = 0; i < n; i++)
      {
	A[i] = ((DATA_TYPE) i+ 2) / n;
	B[i] = ((DATA_TYPE) i+ 3) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
// static
// void print_array(int n,
// 		 DATA_TYPE POLYBENCH_1D(A,N,n))
static
void print_array(int n,
		 DATA_TYPE A[N])
{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    {
      if (i % 20 == 0) printf ("\n");
      printf (DATA_PRINTF_MODIFIER, A[i]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


// /* Main computational kernel. The whole function will be timed,
//    including the call and return. */
// static
// void kernel_jacobi_1d(int tsteps,
// 			    int n,
// 			    DATA_TYPE POLYBENCH_1D(A,N,n),
// 			    DATA_TYPE POLYBENCH_1D(B,N,n))
// {
//   int t, i;

// #pragma scop
//   for (t = 0; t < _PB_TSTEPS; t++)
//     {
//       for (i = 1; i < _PB_N - 1; i++)
// 	      B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
//       for (i = 1; i < _PB_N - 1; i++)
// 	      A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
//     }
// #pragma endscop

// }


int main(int argc, char** argv)
{
  printf("start jacobi-1d large with init and no fence!\n");
  printf("N: %d!\n", N);
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  long long unsigned start;
  long long unsigned end;

  /* Variable declaration/allocation. */
  // POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  // POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);
  DATA_TYPE A[N];
  DATA_TYPE B[N];


  /* Initialize array(s). */
  printf("Initialization\n");
  start = rdcycle();
  init_array (n, A, B);
  end = rdcycle();
  printf("Initialization finished!\n");

  printf("It takes %llu cycles for CPU to finish Initialization.\n", end - start);
  printf("start: %llu\n", start);
  printf("end: %llu\n", end);

  /* Start timer. */
  // polybench_start_instruments;

  /* Run kernel. */
  // kernel_jacobi_1d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
  
  printf("\nstart kernel\n");
  start = rdcycle();
  jacobi_1d(A, B);
  end = rdcycle();

  printf("It takes %llu cycles for CPU to finish the task.\n", end - start);
  printf("start: %llu\n", start);
  printf("end: %llu\n", end);

  /* Stop and print timer. */
  // polybench_stop_instruments;
  // polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  // polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));
  printf("start printing\n"); 
  print_array(n, A);

  /* Be clean. */
  // POLYBENCH_FREE_ARRAY(A);
  // POLYBENCH_FREE_ARRAY(B);

  return 0;
}
