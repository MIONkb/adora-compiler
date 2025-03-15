/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* mvt.c: this file is part of PolyBench/C */

// #include <stdio.h>
// #include <unistd.h>
// #include <string.h>
#include <math.h>

/* Include polybench common header. */
#include "rocket_polybench.h"


/* Include benchmark-specific header. */
#include "mvt.h"
/* Include benchmark-specific header. */
#ifndef ROCKET_TARGET
#define ROCKET_TARGET
#define __riscv
#define __GNUC__
#define MINI_DATASET
#endif
# define MINI_DATASET
# define POLYBENCH_USE_SCALAR_LB
# define POLYBENCH_USE_C99_PROTO
# define DATA_TYPE_IS_FLOAT
#include "include/ISA.h"
#include "adora_test.h"
#include "include/encoding.h"

/* Array initialization. */
// static
// void init_array(int n,
// 		DATA_TYPE POLYBENCH_1D(x1,N,n),
// 		DATA_TYPE POLYBENCH_1D(x2,N,n),
// 		DATA_TYPE POLYBENCH_1D(y_1,N,n),
// 		DATA_TYPE POLYBENCH_1D(y_2,N,n),
// 		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

void    init_array(int n,
		DATA_TYPE x1[N],
		DATA_TYPE x2[N],
		DATA_TYPE y_1[N],
		DATA_TYPE y_2[N],
		DATA_TYPE A[N][N])
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x1[i] = (DATA_TYPE) (i % n) / n;
      x2[i] = (DATA_TYPE) ((i + 1) % n) / n;
      y_1[i] = (DATA_TYPE) ((i + 3) % n) / n;
      y_2[i] = (DATA_TYPE) ((i + 4) % n) / n;
      for (j = 0; j < n; j++)
	A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		//  DATA_TYPE POLYBENCH_1D(x1,N,n),
		//  DATA_TYPE POLYBENCH_1D(x2,N,n))
DATA_TYPE x1[N],
DATA_TYPE x2[N])

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x1");
  for (i = 0; i < n; i++) {
    // if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    // fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x1[i]);
    if (i % 20 == 0) printf ("\n");
    printf(DATA_PRINTF_MODIFIER, x1[i]);
  }
  POLYBENCH_DUMP_END("x1");

  POLYBENCH_DUMP_BEGIN("x2");
  for (i = 0; i < n; i++) {
    // if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    // fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x2[i]);
    if (i % 20 == 0) printf ("\n");
    printf(DATA_PRINTF_MODIFIER, x2[i]);
  }
  POLYBENCH_DUMP_END("x2");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
// static
// void kernel_mvt(int n,
// 		DATA_TYPE POLYBENCH_1D(x1,N,n),
// 		DATA_TYPE POLYBENCH_1D(x2,N,n),
// 		DATA_TYPE POLYBENCH_1D(y_1,N,n),
// 		DATA_TYPE POLYBENCH_1D(y_2,N,n),
// 		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
// {
//   int i, j;

// #pragma scop
//   for (i = 0; i < _PB_N; i++)
//     for (j = 0; j < _PB_N; j++)
//       x1[i] = x1[i] + A[i][j] * y_1[j];
//   for (i = 0; i < _PB_N; i++)
//     for (j = 0; j < _PB_N; j++)
//       x2[i] = x2[i] + A[j][i] * y_2[j];
// #pragma endscop

// }

void kernel_mvt(
		DATA_TYPE x1[N],
		DATA_TYPE x2[N],
		DATA_TYPE y_1[N],
		DATA_TYPE y_2[N],
		DATA_TYPE A[N][N]);


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  printf("CGRA start mvt mini!\n");
//   printf("ldcfg and loaddata is simplified\n");
  printf("N: %d!\n", N);

  int n = N;

  long long unsigned start;
  long long unsigned end;

  /* Variable declaration/allocation. */
  // POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  // POLYBENCH_1D_ARRAY_DECL(x1, DATA_TYPE, N, n);
  // POLYBENCH_1D_ARRAY_DECL(x2, DATA_TYPE, N, n);
  // POLYBENCH_1D_ARRAY_DECL(y_1, DATA_TYPE, N, n);
  // POLYBENCH_1D_ARRAY_DECL(y_2, DATA_TYPE, N, n);
  DATA_TYPE x1[N];
	DATA_TYPE x2[N];
	DATA_TYPE y_1[N];
	DATA_TYPE y_2[N];
	DATA_TYPE A[N][N];

  /* Initialize array(s). */
  // init_array (n,
	//       POLYBENCH_ARRAY(x1),
	//       POLYBENCH_ARRAY(x2),
	//       POLYBENCH_ARRAY(y_1),
	//       POLYBENCH_ARRAY(y_2),
	//       POLYBENCH_ARRAY(A));
  init_array(n ,x1, x2, y_1, y_2, A);

  /* Start timer. */
  // polybench_start_instruments;

  printf("\nstart kernel\n");
  start = rdcycle();
  /* Run kernel. */
  // kernel_mvt (n,
	//       POLYBENCH_ARRAY(x1),
	//       POLYBENCH_ARRAY(x2),
	//       POLYBENCH_ARRAY(y_1),
	//       POLYBENCH_ARRAY(y_2),
	//       POLYBENCH_ARRAY(A));
  kernel_mvt(x1, x2, y_1, y_2, A);

    fence(1);
  end = rdcycle();

  /* Stop and print timer. */
  // polybench_stop_instruments;
  // polybench_print_instruments;
  printf("It takes %llu cycles for CPU to finish the task.\n", end - start);
  printf("start: %llu\n", start);
  printf("end: %llu\n", end);
  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  // polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2)));

  print_array(n, x1, x2);
  /* Be clean. */
  // POLYBENCH_FREE_ARRAY(A);
  // POLYBENCH_FREE_ARRAY(x1);
  // POLYBENCH_FREE_ARRAY(x2);
  // POLYBENCH_FREE_ARRAY(y_1);
  // POLYBENCH_FREE_ARRAY(y_2);

  return 0;
}
