// #include <stdio.h>
// #include <unistd.h>
// #include <string.h>
// #include <math.h>
// #include <malloc.h>
#include "include/encoding.h"
#include "include/ISA.h"


float* forward(float[128][64]);
void *malloc (size_t sz);
//return %alloc_11 : memref<1x64x56x56xf32>
int main(int argc, char** argv)
{
  printf("CPU+cgra execute relu 128 64!\n");
  // printf("mem copy is removed\n");
  // printf("CGRA instructions are simplified!\n");
  long long unsigned start;
  long long unsigned end;
	float a [128][64];


	int d0 , d1, d2, d3;
	for(d0 = 0; d0 < 128; d0++){
    for(d1 = 0; d1 < 64; d1++){
      a[d0][d1]=1;
    }
  } 
  printf("Start kernel!\n");

  start = rdcycle();
  float* b = (float*)forward(a);
  end = rdcycle();
  printf("It takes %d cycles for cpu to finish the task.\n", end - start);
  printf("start cycle:%d\n",start);
  printf("c:\n");
	for(int i = 0; i < 1*128*64; i++){
    //return %alloc_11 : memref<1x64x56x56xf32>
      int I = (int)(*(b+i)*10000);
      printf("b[%d]:%d\n",i, I);
  } 
  // result_check();
  printf("test complete!\n");

  return 0;
}