#include <math.h>
/* Include polybench common header. */
#include "rocket_polybench.h"

/* Include benchmark-specific header. */
#include "3mm.h"
#include "adora_test.h"
// void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
// 		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
// 		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
// 		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
// 		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
// 		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
// 		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
// 		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
void kernel_3mm(
		DATA_TYPE E[NI][NJ],
		DATA_TYPE A[NI][NK],
		DATA_TYPE B[NK][NJ],
		DATA_TYPE F[NJ][NL],
		DATA_TYPE C[NJ][NM],
		DATA_TYPE D[NM][NL],
		DATA_TYPE G[NI][NL])
{
  int i, j, k;

#pragma scop
  /* E := A*B */
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      {
	E[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < NK; ++k)
	  E[i][j] += A[i][k] * B[k][j];
      }

  /* F := C*D */
  for (i = 0; i < NJ; i++)
    for (j = 0; j < NL; j++)
      {
	F[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < NM; ++k)
	  F[i][j] += C[i][k] * D[k][j];
      }
      
  /* G := E*F */
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++)
      {
	G[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < NJ; ++k)
	  G[i][j] += E[i][k] * F[k][j];
      }
#pragma endscop

}