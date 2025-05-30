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
// #include <math.h>

/* Include polybench common header. */
// #include "rocket_polybench.h"

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

#   define W 64
#   define H 64

#include "encoding.h"

#include "ISA.h"

#define FpToHex(x) *((unsigned int*)&x)
void kernel_deriche(void* arg_0 ,void* arg_1){
  float float_10;
  float_10 = 0;
  float float_11;
  float_11 = 11;
  float float_12;
  float_12 = 12;
  float float_13;
  float_13 = 13;

  
  uint8_t _task_id = 0;
  for (int int_14 = 0; int_14 < 64; int_14 = int_14 + 32){
    /// %1 = ADORA.BlockLoad %arg0 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_0"}
    // uint64_t dramoffset_0 = 256 * int_14;
    uint64_t dramoffset_0 = 0 * int_14;
    load_data(arg_0 + dramoffset_0 , 0x18000, 8192, 0, _task_id, 0);


    load_data(&float_11, 0x8000, 8, 0, _task_id, 0);

    load_data(&float_10, 0xa000, 8, 0, _task_id, 0);

    load_data(&float_13, 0x0, 8, 0, _task_id, 0);

    fence(1);
    
    
    // /// ADORA.BlockStore %5, %alloca_12 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_0"}
    // store(&float_11, 0x8000, 8, _task_id, 0);

    
    // /// ADORA.BlockStore %4, %alloca_11 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_0"}
    // store(&float_10, 0xa000, 8, _task_id, 0);

    
    
    // /// ADORA.BlockStore %3, %alloca_14 [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_0"}
    // store(&float_13, 0x0, 8, _task_id, 0);

    
    
    /// ADORA.BlockStore %2, %arg2 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_0"}
    // uint64_t dramoffset_1 = 256 * int_14;

    /// kernel_deriche_1
    volatile unsigned short cin[78][3] __attribute__((aligned(8))) = { //// 8301_4ed8

												                //// 8301_4eC0
    	{0x1000, 0x0000, 0x0008},{0x0001, //// 8301_4ed8
			0x0100, 0x0009},{0x0000, 0x0000,  ////
			0x000a},{0x0000, 0x086a, 0x000b},

    	{0x0020, 0x0000, 0x000c},{0x0800, //// 8301_4e80
			0x0000, 0x0010},{0x0001, 0x0100,  //// 8301_4e88
			0x0011},{0x0000, 0x0000, 0x0012}, 
    	{0x0000, 0x08ea, 0x0013},{0x0020, 
			0x0000, 0x0014},{0xe03F, 0x03ff,
			0x0018},{0x1fc1, 0x0100, 0x0019},
    		{0x0000, 0x0000, 0x001a},{0x0000, /// 8_0000_0002
			0x000a, 0x001b},{0xf83f, 0x03ff,    /// 9_0000_0001

			0x0020},{0x1fc1, 0x0100, 0x0021},	  //// 8301_4fC0
    		{0x0000, 0x0000, 0x0022},{0x0000, ///
			0x08ea, 0x0023},{0x0000, 0x0000, 	  ///  
			0x0024},{0x0000, 0x0000, 0x0028},

			/////
    		{0x0001, 0x0100, 0x0029},
    		{0x0000, 0x0000, 0x002a},
    		{0x0000, 0x084a, 0x002b},
    		{0x0000, 0x0000, 0x002c},
    		{0x0003, 0x0000, 0x0058},
    		{0x0000, 0x0010, 0x0060},
    		{0x0010, 0x0000, 0x0068},
    		{0x0030, 0x0000, 0x0070},
    		{0x0150, 0x0020, 0x0099},
    		{0x0000, 0x5000, 0x009a},
    		{0x0000, 0x8010, 0x009b},
    		{0x0040, 0x0000, 0x009c},
    		{0x44fd, 0x3f57, 0x00a0},
    		{0x0008, 0x0100, 0x00a1},
    		{0x018a, 0x0118, 0x00a9},
    		{0x0080, 0x0008, 0x00b1},
    		{0x0000, 0x0000, 0x00e0},
    		{0x0000, 0x2000, 0x00e8},
    		{0x0084, 0x0000, 0x00f0},
    		{0x0003, 0x0000, 0x00f1},
    		{0x0004, 0x0004, 0x00f8},
    		{0x0000, 0x000c, 0x0100},
    		{0x0000, 0x0080, 0x0108},
    		{0x4598, 0xbf1b, 0x0128},
    		{0x0008, 0x0040, 0x0129},
    		{0x0050, 0x0010, 0x0131},
    		{0x0000, 0x5000, 0x0132},
    		{0x0200, 0x8010, 0x0133},
    		{0x0040, 0x0000, 0x0134},
    		{0x008a, 0x0050, 0x0139},
    		{0x000a, 0x0110, 0x0141},
    		{0x0050, 0x0008, 0x0149},
    		{0x0000, 0x5000, 0x014a},
    		{0x0000, 0x8010, 0x014b},
    		{0x0040, 0x0000, 0x014c},
    		{0x6028, 0x3dea, 0x0150},
    		{0x0008, 0x0018, 0x0151},
    		{0x0003, 0x0000, 0x0181},
    		{0x0000, 0x8000, 0x0188},
    		{0x1004, 0x0000, 0x0190},
    		{0x0001, 0x0000, 0x0191},
    		{0x0040, 0x0000, 0x0198},
    		{0x1714, 0xbe3c, 0x01d0},
    		{0x0008, 0x0010, 0x01d1},
    		{0x0050, 0x0008, 0x01d9},
    		{0x0000, 0x5000, 0x01da},
    		{0x0000, 0x8010, 0x01db},
    		{0x0040, 0x0000, 0x01dc},
    		{0x0003, 0x0000, 0x0211},
    		{0x0000, 0x0c00, 0x0218},
    		{0x2000, 0x0080, 0x0220},
    		{0x0080, 0x0008, 0x0269},
    		{0x2000, 0x0000, 0x02a0},
    		{0x0000, 0x0000, 0x02e0},
    		{0x0001, 0x0100, 0x02e1},
    		{0x0000, 0x0000, 0x02e2},
    		{0x0000, 0x08aa, 0x02e3},
    		{0x0020, 0x0000, 0x02e4},
    	};

	// printf("cin: %x\n", cin);
	// for(int i = 0; i < 78; i++){
	// 	printf("cin[%d]: %x, %x, %x\n",i, cin[i][0], cin[i][1], cin[i][2]);
	// }
    
    load_cfg((void*)cin, 0x20000, 468, _task_id, 0);

    uint64_t dramoffset_1 = 0 * int_14;
    store(arg_1 + dramoffset_1 , 0x18000 , 256, _task_id, 0);

    store(arg_1 + dramoffset_1 + 256, 0x18000 + 256, 256, _task_id, 0);

    store(arg_1 + dramoffset_1 + 512, 0x18000 + 512, 256, _task_id, 0);

    store(arg_1 + dramoffset_1 + 512 + 256, 0x18000 + 512 + 256, 256, _task_id, 0);

    store(arg_1 + dramoffset_1 + 1024 + 256, 0x18000 + 1024 + 256, 256, _task_id, 0);

    store(arg_1 + dramoffset_1 + 1024 + 256, 0x18000 + 1024 + 256, 256, _task_id, 0);

    // fence(1);
    printf(" complete for one time!\n");
    _task_id++;
  }
}



/* Array initialization. */
void init_array (int w, int h,
		 float imgIn[W][H],
		 float imgOut[W][H])
{
  int i, j;
  w = W;
  h = H;

  //input should be between 0 and 1 (grayscale image pixel)
  for (i = 0; i < w; i++)
     for (j = 0; j < h; j++)
	    imgIn[i][j] = (float) ((313*i+991*j)%65536) / 65535.0f;

  for (j = 0; j < 3; j++)
	  printf("%x ",FpToHex(imgIn[0][j]));
  printf("\n");
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_image1and2(int w, int h,
		 /*DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h)*/float imgOut1[W][H], float imgOut2[W][H])

{
  int i, j;
  // for (i = 0; i < w; i++)
  //   for (j = 0; j < h; j++) {

  for (i = 0; i < 5; i++)
    for (j = 0; j < 5; j++) {
      if ((i * h + j) % 20 == 0) printf("\n");
      printf("[%d, %d]%ld-%ld," , i, j, (int)(10000 * imgOut1[i][j]), (int)(10000 * imgOut2[i][j]));
      printf("%x-%x\n", FpToHex(imgOut1[i][j]), FpToHex(imgOut2[i][j]));
    }
}

int main(int argc, char** argv)
{
  printf("start DMA test!\n");
//   printf("ldcfg and loaddata is simplified\n");
  printf("W: %d!\n", W);
  printf("H: %d!\n", H);
  /* Retrieve problem size. */
  int w = W;
  int h = H;

  long long unsigned start;
  long long unsigned end;

  /* Variable declaration/allocation. */
  float alpha;

  float imgIn[W][H];
  float imgOut[W][H];
  float y1_1[W][H], y1_2[W][H];
  float y2[W][H];

  /* Initialize array(s). */
  init_array (w, h, imgIn, imgOut);
  /* Start timer. */

  /* Run kernel. */
  printf("imgIn addr: %x!\n", &imgIn);
  printf("imgOut addr: %x!\n", &imgOut);

  kernel_deriche(imgIn, imgOut);

  print_image1and2(W, H, imgIn, imgOut);


  return 0;
}
