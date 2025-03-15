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
#include <math.h>

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
#include "gesummv.h"

#include "adora_test.h"
#include "include/encoding.h"

/* Array initialization. */
// static
// void init_array(int n,
// 		DATA_TYPE *alpha,
// 		DATA_TYPE *beta,
// 		DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
// 		DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
// 		DATA_TYPE POLYBENCH_1D(x,N,n))

static
void init_array(
		DATA_TYPE A[N][N],
		DATA_TYPE B[N][N],
		DATA_TYPE x[N])
{
  int i, j;
  int n = N;
  // *alpha = 1.5;
  // *beta = 1.2;
  for (i = 0; i < n; i++)
    {
      x[i] = (DATA_TYPE)( i % n) / n;
      for (j = 0; j < n; j++) {
	A[i][j] = (DATA_TYPE) ((i*j+1) % n) / n;
	B[i][j] = (DATA_TYPE) ((i*j+2) % n) / n;
      }
    }
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
      printf (DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
// static
// void kernel_gesummv(int n,
// 		    DATA_TYPE alpha,
// 		    DATA_TYPE beta,
// 		    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
// 		    DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
// 		    DATA_TYPE POLYBENCH_1D(tmp,N,n),
// 		    DATA_TYPE POLYBENCH_1D(x,N,n),
// 		    DATA_TYPE POLYBENCH_1D(y,N,n))
// {
//   int i, j;

// #pragma scop
//   for (i = 0; i < _PB_N; i++)
//     {
//       tmp[i] = SCALAR_VAL(0.0);
//       y[i] = SCALAR_VAL(0.0);
//       for (j = 0; j < _PB_N; j++)
// 	{
// 	  tmp[i] = A[i][j] * x[j] + tmp[i];
// 	  y[i] = B[i][j] * x[j] + y[i];
// 	}
//       y[i] = alpha * tmp[i] + beta * y[i];
//     }
// #pragma endscop

// }

void gesummv(
		    DATA_TYPE A[N][N] ,
		    DATA_TYPE B[N][N] ,
		    DATA_TYPE tmp[N] ,
		    DATA_TYPE x[N] ,
		    DATA_TYPE y[N] );

int main(int argc, char** argv)
{

  printf("CGRA start gesummv medium without init!\n");
  printf("ldcfg and loaddata is simplified\n");
  printf("N: %d!\n", N);
  /* Retrieve problem size. */
  int n = N;

  long long unsigned start;
  long long unsigned end;

  /* Variable declaration/allocation. */
  // DATA_TYPE alpha;
  // DATA_TYPE beta;
  // POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  // POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  // POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, N, n);
  // POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  // POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
	DATA_TYPE A[N][N] ;
	DATA_TYPE B[N][N] ;
	DATA_TYPE tmp[N] ;
	DATA_TYPE x[N] ;
	DATA_TYPE y[N] ;

  /* Initialize array(s). */
  // init_array (n, &alpha, &beta,
	//       POLYBENCH_ARRAY(A),
	//       POLYBENCH_ARRAY(B),
	//       POLYBENCH_ARRAY(x));
  // printf("Initialization\n");
  // start = rdcycle();
  // init_array(A, B ,x);
  // end = rdcycle();
  // printf("Initialization finished!\n");
  // printf("It takes %llu cycles for CPU to finish Initialization.\n", end - start);
 

  /* Start timer. */
  // polybench_start_instruments;

  /* Run kernel. */
  // kernel_gesummv (n, alpha, beta,
	// 	  POLYBENCH_ARRAY(A),
	// 	  POLYBENCH_ARRAY(B),
	// 	  POLYBENCH_ARRAY(tmp),
	// 	  POLYBENCH_ARRAY(x),
	// 	  POLYBENCH_ARRAY(y));

  printf("\nstart kernel\n");
  start = rdcycle();
  gesummv(A, B, tmp, x, y);
  end = rdcycle();

  printf("It takes %llu cycles for CPU to finish the task.\n", end - start);
  printf("start: %llu\n", start);
  printf("end: %llu\n", end);

  /* Stop and print timer. */
  // polybench_stop_instruments;
  // polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */

  printf("start printing\n"); 
  print_array(n, y);

  // polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  // /* Be clean. */
  // POLYBENCH_FREE_ARRAY(A);
  // POLYBENCH_FREE_ARRAY(B);
  // POLYBENCH_FREE_ARRAY(tmp);
  // POLYBENCH_FREE_ARRAY(x);
  // POLYBENCH_FREE_ARRAY(y);

  return 0;
}
