#ifndef ROCKET_TARGET
#define ROCKET_TARGET
#define __riscv
#define __GNUC__
#endif
# define POLYBENCH_USE_SCALAR_LB
# define POLYBENCH_USE_C99_PROTO
# define DATA_TYPE_IS_FLOAT
#include <math.h>
#include "cholesky.h"


/* Include polybench common header. */
// #include <polybench.h>
#include "rocket_polybench.h"

#include "cgrv_test.h"
#include "include/encoding.h"

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
// void cholesky(int n,
// 		     DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
void cholesky(DATA_TYPE A[N][N])
{
  int i, j, k;


#pragma scop
  for (i = 0; i < _PB_N; i++) {
     //j<i
     for (j = 0; j < i; j++) {
        for (k = 0; k < j; k++) {
           A[i][j] -= A[i][k] * A[j][k];
        }
        A[i][j] /= A[j][j];
     }
     // i==j case
     for (k = 0; k < i; k++) {
        A[i][i] -= A[i][k] * A[i][k];
     }
     A[i][i] = SQRT_FUN(A[i][i]);
  }
#pragma endscop

}