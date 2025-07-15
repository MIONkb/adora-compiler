#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include "include/encoding.h"
#include "include/ISA.h"

#define size 4

// float* forward(float[3][3]);
void kernel_merge4(int a[size], int b[size], int c[size], int d[size], int r[4*size]);

int main(int argc, char** argv)
{
  printf("CGRA execute kernel_merge4!\n");
  long long unsigned start, cur;
  long long unsigned end;
	// int a [500][10000], b[500];
  // printf("a addr:%x\n", a);
  // printf("b addr:%x\n", b);

	int i , j;

	// for(i = 0; i < 500; i++){
  //   b[i] = i * 2;
  //   for(j = 0; j < 10; j++){
	// 	  a[i][j]=i + j;
  //   }
  // } 
  // printf("start cycle 0:%d\n",start);
  // for(i = 0; i < 400; i++){
  //   for(j = 0; j < 100; j++){
	// 	  a[i][j]= a[i][j] * b[i];
  //   }
  // } 

  // cur = rdcycle();
  // printf("cur cycle 1:%d\n",cur);
  // float* b = (float*)forward(a);
  // end = rdcycle();
  int a[size], b[size], c[size], d[size];
  int r[size * 4];


  // int * na = (int *)malloc(8196 * sizeof(int));
  // printf("na addr:%x\n", na);
  start = rdcycle();
  for(i = 0; i < size; i++){
		  a[i]= i * 4 ;
		  b[i]= i * 4 + 1 ;
		  c[i]= i * 4 + 2 ;
		  d[i]= i * 4 + 3 ;
  } 
  end = rdcycle();
  printf("It takes %d cycles for CPU to finish the initialization.\n", end - start);

  start = rdcycle();
  kernel_merge4(a, b, c, d, r);
  fence(1);
  end = rdcycle();
  printf("It takes %d cycles for CGRA to finish the task.\n", end - start);

  for ( int k=0 ; k<size*4 ; k++ ) {
    printf("[%d]%d,", k, r[k]);
  }      


  start = rdcycle();
  for(i = 0; i < size; i++){
		  r[i]= a[i];
		  r[i * 4 + 1]= b[i];
		  r[i * 4 + 2]= c[i];
		  r[i * 4 + 3]= d[i];
  } 
  end = rdcycle();
  printf("It takes %d cycles for CPU to finish the initialization.\n", end - start);

  printf("test complete!\n");

  return 0;
}