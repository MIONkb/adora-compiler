
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void cholesky_kernel_1(void* arg_0 ,int arg_1){
    printf("Start kernel 1\n");
    /// %0 = ADORA.BlockLoad %arg0 [%arg1, %arg1] : memref<120x120xf32> -> memref<1x2xf32>  {Id = "0", KernelName = "cholesky_kernel_1"}
    uint64_t dramoffset_0 = 480 * arg_1 + 4 * arg_1;
    uint64_t spadoffset_0 = 0;
    for(int idx_0 = 0; idx_0 < 1; idx_0++){
      uint64_t roffset_0 =  480*idx_0 ;
      load_data(arg_0 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 8, 0, 0, 0);
      spadoffset_0 = spadoffset_0 + 8;
    } 

    printf("Finish first load\n");

    /// %1 = ADORA.BlockLoad %arg0 [%arg1, 0] : memref<120x120xf32> -> memref<1x120xf32>  {Id = "1", KernelName = "cholesky_kernel_1"}
    uint64_t dramoffset_1 = 480 * arg_1;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_0 + dramoffset_1 + roffset_1, 0x0 + spadoffset_1, 480, 0, 0, 0);
    spadoffset_1 = spadoffset_1 + 480;
    printf("Finish second load\n");
    
    int int_2 = arg_1;
    int int_3 = arg_1;
    int int_4 = arg_1;
    int int_5 = arg_1;
    volatile unsigned short cin[24][3] __attribute__((aligned(8))) = {
    		{0x2000, 0xf800, 0x0010},
    		{0x0013, 0x0000, 0x0011},
    		{0x0000, 0x0100, 0x0012},
    		{0x0000, 0x0000, 0x0013},
    		{0x0800, 0xf800, 0x0020},
    		{0x0013, 0x0000, 0x0021},
    		{0x0000, 0x8f00, 0x0022},
    		{0x0000, 0x0000, 0x0023},
    		{0x0000, 0xf800, 0x0030},
    		{0x0013, 0x0000, 0x0031},
    		{0x0000, 0x0100, 0x0032},
    		{0x0000, 0x0000, 0x0033},
    		{0x0000, 0x0000, 0x0058},
    		{0x8010, 0x0000, 0x0068},
    		{0x0000, 0x0002, 0x0070},
    		{0x0000, 0x0000, 0x0078},
    		{0x000d, 0x0004, 0x0099},
    		{0x040e, 0x0026, 0x00a9},
    		{0x0000, 0x0000, 0x00e8},
    		{0x0040, 0x0000, 0x00f0},
    		{0x0010, 0x0002, 0x0131},
    		{0x0000, 0x0400, 0x0132},
    		{0xe140, 0x014f, 0x0133},
    		{0x0000, 0x0000, 0x0134},
    	};
    
    cin[0][1] = (int_2 << 0xa) | (cin[0][1] & 0x3ff);
    cin[1][0] = (int_2 >> 0x6) | (cin[1][0] & 0xfc00);
    cin[8][1] = (int_3 << 0xa) | (cin[8][1] & 0x3ff);
    cin[9][0] = (int_3 >> 0x6) | (cin[9][0] & 0xfc00);
    cin[4][1] = (int_4 << 0xa) | (cin[4][1] & 0x3ff);
    cin[5][0] = (int_4 >> 0x6) | (cin[5][0] & 0xfc00);
    cin[22][0] = (int_5 << 0xc) | (cin[22][0] & 0xfff);
    cin[22][1] = (int_5 >> 0x8) | (cin[22][1] & 0xff00);
    load_cfg((void*)cin, 0x20000, 144, 0, 0);
    config(0x0, 24, 0, 0);

    printf("Finish cfg\n");


    execute(0x2a, 0, 0);
     printf("Finish exe\n");
    /// ADORA.BlockStore %2, %arg0 [%arg1, %arg1] : memref<1x2xf32> -> memref<120x120xf32>  {Id = "2", KernelName = "cholesky_kernel_1"}
    uint64_t dramoffset_2 = 8 * arg_1 + 4 * arg_1;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_0 + dramoffset_2 + roffset_2, 0x2000 + spadoffset_2, 8, 0, 0);
    spadoffset_2 = spadoffset_2 + 8;
    
     printf("Finish kernel1\n");
}
