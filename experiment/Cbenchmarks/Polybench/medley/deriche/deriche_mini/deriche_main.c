/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* deriche.c: this file is part of PolyBench/C */

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
#define MINI_DATASET
#endif
# define MINI_DATASET
# define POLYBENCH_USE_SCALAR_LB
# define POLYBENCH_USE_C99_PROTO
# define DATA_TYPE_IS_FLOAT

#include "deriche.h"
#include "adora_test.h"
#include "include/encoding.h"

#define FpToHex(x) *((unsigned int*)&x)

/* Array initialization. */
// static
// void init_array (int w, int h, DATA_TYPE* alpha,
// 		 DATA_TYPE POLYBENCH_2D(imgIn,W,H,w,h),
// 		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))
void init_array (int w, int h,
		 DATA_TYPE imgIn[W][H],
		 DATA_TYPE imgOut[W][H])
{
  int i, j;
  w = W;
  h = H;
//   *alpha=0.25; //parameter of the filter

  //input should be between 0 and 1 (grayscale image pixel)
  for (i = 0; i < w; i++)
     for (j = 0; j < h; j++)
	    imgIn[i][j] = (DATA_TYPE) ((313*i+991*j)%65536) / 65535.0f;

  for (j = 0; j < 3; j++)
	  printf("%x ",FpToHex(imgIn[0][j]));
  printf("\n");
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_image1and2(int w, int h,
		 /*DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h)*/DATA_TYPE imgOut1[W][H], DATA_TYPE imgOut2[W][H])

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("imgOut");
  // for (i = 0; i < w; i++)
  //   for (j = 0; j < h; j++) {

  for (i = 0; i < w; i=i+1){
    for (j = 0; j < h; j = j + 1) {
    //   if ((i * h + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
    //   fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, imgOut[i][j]);
      // if ((i * h + j) % 20 == 0) printf("\n");
      // printf(DATA_PRINTF_MODIFIER, imgOut[i][j]);
      if((int)(10000 * imgOut1[i][j]) != (int)(10000 * imgOut2[i][j])){
        printf("[%d, %d]%ld-%ld," , i, j, (int)(10000 * imgOut1[i][j]), (int)(10000 * imgOut2[i][j]));
        printf("%x-%x\t", FpToHex(imgOut1[i][j]), FpToHex(imgOut2[i][j]));
      }
    }
    printf("over one line:[%d, %d]%x-%x" , i, 0, FpToHex(imgOut1[i][0]), FpToHex(imgOut2[i][0]));
    printf("\n");
  }
  POLYBENCH_DUMP_END("imgOut");
  POLYBENCH_DUMP_FINISH;
}

void kernel_deriche(
       DATA_TYPE imgIn[W][H] ,
       DATA_TYPE imgOut[W][H] ,
       DATA_TYPE y1[W][H] ,
       DATA_TYPE y2[W][H] );

void cpu_kernel_deriche(
       DATA_TYPE imgIn[W][H] ,
       DATA_TYPE imgOut[W][H] ,
       DATA_TYPE y1[W][H] ,
       DATA_TYPE y2[W][H] );

int main(int argc, char** argv)
{
  printf("CGRA start deriche mini!\n");
//   printf("ldcfg and loaddata is simplified\n");
  printf("W: %d!\n", W);
  printf("H: %d!\n", H);
  /* Retrieve problem size. */
  int w = W;
  int h = H;

  long long unsigned start;
  long long unsigned end;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
//   POLYBENCH_2D_ARRAY_DECL(imgIn, DATA_TYPE, W, H, w, h);
//   POLYBENCH_2D_ARRAY_DECL(imgOut, DATA_TYPE, W, H, w, h);
//   POLYBENCH_2D_ARRAY_DECL(y1, DATA_TYPE, W, H, w, h);
//   POLYBENCH_2D_ARRAY_DECL(y2, DATA_TYPE, W, H, w, h);
  DATA_TYPE imgIn[W][H];
  DATA_TYPE imgOut1[W][H], imgOut2[W][H];
  DATA_TYPE y1_1[W][H], y1_2[W][H];
  DATA_TYPE y2_1[W][H], y2_2[W][H];

  /* Initialize array(s). */
  // init_array (w, h, &alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut));
  init_array (w, h, imgIn, imgOut1);
  /* Start timer. */
//   polybench_start_instruments;

  /* Run kernel. */
//   kernel_deriche (w, h, alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut), POLYBENCH_ARRAY(y1), POLYBENCH_ARRAY(y2));
  printf("imgIn: %x!\n", &imgIn);
  printf("imgOut: %x!\n", &imgOut1);
  printf("\nstart kernel\n");
  start = rdcycle();
  kernel_deriche(imgIn, imgOut1, y1_1, y2_1);
  end = rdcycle();

  printf("It takes %llu cycles for CGRA to finish the task.\n", end - start);
  printf("start: %llu\n", start);
  printf("end: %llu\n", end);

  // print_array(W, H, imgOut);

  init_array (w, h, imgIn, imgOut2);
  printf("\nstart cpu kernel\n");
  start = rdcycle();
  cpu_kernel_deriche(imgIn, imgOut2, y1_2, y2_2);
  
  end = rdcycle();

  printf("It takes %llu cycles for cpu to finish the task.\n", end - start);
  printf("start: %llu\n", start);
  printf("end: %llu\n", end);
//   /* Stop and print timer. */
//   polybench_stop_instruments;
//   polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
//   polybench_prevent_dce(print_array(w, h, POLYBENCH_ARRAY(imgOut)));
    // print_image1and2(W, H, y1_2, y1_2);
    printf("imgOut1\n");
    // print_image1and2(W, H, imgIn, imgIn);
    // print_image1and2(W, H, y1_1, y1_2);
    print_image1and2(W, H, imgOut1, imgOut2);

    printf("y1\n");
    // // print_image1and2(W, H, imgIn, imgIn);
    // // print_image1and2(W, H, y1_1, y1_2);
    print_image1and2(W, H, y1_1, y1_2);

    printf("y2\n");
    // // print_image1and2(W, H, imgIn, imgIn);
    // // print_image1and2(W, H, y1_1, y1_2);
    print_image1and2(W, H, y2_1, y2_2);
  /* Be clean. */
//   POLYBENCH_FREE_ARRAY(imgIn);
//   POLYBENCH_FREE_ARRAY(imgOut);
//   POLYBENCH_FREE_ARRAY(y1);
//   POLYBENCH_FREE_ARRAY(y2);

  return 0;
}
