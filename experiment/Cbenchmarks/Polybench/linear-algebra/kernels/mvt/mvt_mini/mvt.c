#include <math.h>
/* Include polybench common header. */
#include "rocket_polybench.h"

/* Include benchmark-specific header. */
#include "mvt.h"
#include "adora_test.h"

// void kernel_mvt(int n,
// 		DATA_TYPE POLYBENCH_1D(x1,N,n),
// 		DATA_TYPE POLYBENCH_1D(x2,N,n),
// 		DATA_TYPE POLYBENCH_1D(y_1,N,n),
// 		DATA_TYPE POLYBENCH_1D(y_2,N,n),
// 		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
// {
void kernel_mvt(
		DATA_TYPE x1[N],
		DATA_TYPE x2[N],
		DATA_TYPE y_1[N],
		DATA_TYPE y_2[N],
		DATA_TYPE A[N][N])
{
  int i, j;

#pragma scop
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];
#pragma endscop

}