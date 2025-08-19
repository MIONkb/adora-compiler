#include <math.h>

# define DATA_TYPE_IS_FLOAT

/* Include polybench common header. */
#include "rocket_polybench.h"

/* Include benchmark-specific header. */
#include "jacobi-1d.h"
#include "adora_test.h"


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
// void jacobi_1d(int tsteps,
// 			    int n,
// 			    DATA_TYPE POLYBENCH_1D(A,N,n),
// 			    DATA_TYPE POLYBENCH_1D(B,N,n))

void jacobi_1d(
          DATA_TYPE A[N],
			    DATA_TYPE B[N])
{
  int t, i;

#pragma scop
  for (t = 0; t < TSTEPS; t++)
    {
      for (i = 1; i < N - 1; i++)
	      B[i] = (float)0.33333 * (A[i-1] + A[i] + A[i + 1]);
      for (i = 1; i < N - 1; i++)
	      A[i] = (float)0.33333 * (B[i-1] + B[i] + B[i + 1]);
    }
#pragma endscop

}
