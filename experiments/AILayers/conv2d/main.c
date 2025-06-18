// #include <stdio.h>
// #include <unistd.h>
// #include <string.h>
// #include <math.h>
// #include <malloc.h>
#include "include/encoding.h"
#include "include/ISA.h"


float* conv2d(float[1][3][64][64]);
void *malloc (size_t sz);
//return %alloc_11 : memref<1x64x56x56xf32>
int main(int argc, char** argv)
{
  printf("CPU+cgra execute conv 3 64 64 no memcpy with fence after store!\n");
  printf("CPU compute first part\n");
  printf("memcpy enabled\n");
  long long unsigned start;
  long long unsigned end;
	float a [1][3][64][64];

  printf("Before malloc test \n");
   float* c = (float*) malloc(64*sizeof(float));
  printf("end malloc test! \n");
  printf("c: %x\n" ,c);

	int d0 , d1, d2, d3;
	for(d0 = 0; d0 < 1; d0++){
    for(d1 = 0; d1 < 3; d1++){
      for(d2 = 0; d2 < 64; d2++){
        for(d3 = 0; d3 < 64; d3++){
          a[d0][d1][d2][d3]=1;
        }
      }
    }
  } 
  printf("value assign!\n");

  start = rdcycle();
  float* b = (float*)conv2d(a);
  end = rdcycle();
  printf("It takes %d cycles for cpu to finish the task.\n", end - start);
  printf("start cycle:%d\n",start);
  printf("c:\n");
	for(int i = 0; i < 1*64*56*56; i++){
    //return %alloc_11 : memref<1x64x56x56xf32>
      int I = (int)(*(b+i)*10000);
      printf("b[%d]:%d\n",i, I);
  } 
  // result_check();
  printf("test complete!\n");

  return 0;
}