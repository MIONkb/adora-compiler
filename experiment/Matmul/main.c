#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include "include/encoding.h"
#include "include/ISA.h"


#include "Matmul.h"

static
void print_2_2DMatrix(Dtype r0[W][H], Dtype r1[W][H])

{
  int i, j;
   printf("Start printing.\n");

  for (i = 0; i < W; i = i+1){
    for (j = 0; j < H; j = j + 1) {
      printf("[%d, %d]%x-%x," , i, j, r0[i][j], r1[i][j]);
      // if((int)(10000 * imgOut1[i][j]) != (int)(10000 * imgOut2[i][j])){
      //   printf("[%d, %d]%ld-%ld," , i, j, (int)(10000 * imgOut1[i][j]), (int)(10000 * imgOut2[i][j]));
      //   printf("%x-%x\t", FpToHex(imgOut1[i][j]), FpToHex(imgOut2[i][j]));
      // }
    }
    // printf("over one line:[%d, %d]%x-%x" , i, 0, FpToHex(imgOut1[i][0]), FpToHex(imgOut2[i][0]));
    printf("\n");
  }
}


void initialize_abc(Dtype a[W][N], Dtype b[N][H], Dtype c[W][H]) {
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = i;  
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < H; j++) {
            b[i][j] = j + 10; 
        }
    }

    for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++) {
            c[i][j] = i + 100; 
        }
    }
}

void initialize_r(Dtype r[W][H]) {
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++) {
            r[i][j] = 0;  
        }
    }
}

void cpu_MATMUL(Dtype a[W][N], Dtype b[N][H], Dtype c[W][H], Dtype r[W][H]) {
    #pragma scop
    for ( int i=0 ; i<W ; i++ ) {
        for(int j=0 ; j<H ; j++){
            r[i][j] = c[i][j];
            for(int k=0 ; k<W ; k++){
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }      
    #pragma endscop
}

int main(int argc, char** argv)
{
  printf("CGRA execute merge_MATMUL and unroll_MATMUL!\n");
  long long unsigned start, cur;
  long long unsigned end;

	int i , j;

  start = rdcycle();
  Dtype a[W][N]; Dtype b[N][H]; Dtype c[W][H];
  Dtype r0[W][H]; Dtype r1[W][H] ; Dtype r2[W][H];
  initialize_abc(a, b, c);
  initialize_r(r0);
  initialize_r(r1);
  initialize_r(r2);
  end = rdcycle();
  printf("It takes %d cycles for CPU to finish the initialization.\n", end - start);

  printf("a: %x, b:%x, c: %x\n", &a, &b, &c);

  start = rdcycle();
  cpu_MATMUL(a, b, c, r0);
  end = rdcycle();
  printf("It takes %d cycles for CPU to finish the task.\n", end - start);

  start = rdcycle();
  unroll_MATMUL(a, b, c, r1);
  end = rdcycle();
  printf("It takes %d cycles for CGRA to finish the unroll_MATMUL task.\n", end - start);

  start = rdcycle();
  merge_MATMUL(a, b, c, r2);
  end = rdcycle();
  printf("It takes %d cycles for CGRA to finish the merge_MATMUL task.\n", end - start);

  printf("Start print r0 r1\n");
  print_2_2DMatrix(r0, r1);

  printf("Start print r0 r2\n");
  print_2_2DMatrix(r0, r2);

  printf("test complete!\n");


  return 0;
}