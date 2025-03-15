
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void cholesky_kernel_1(void* arg_0 ,int arg_1){
    /// %0 = ADORA.BlockLoad %arg0 [%arg1, %arg1] : memref<2000x2000xf32> -> memref<1x2xf32>  {Id = "0", KernelName = "cholesky_kernel_1"}
    uint64_t dramoffset_0 = 8000 * arg_1 + 4 * arg_1;
    uint64_t spadoffset_0 = 0;
    for(int idx_0 = 0; idx_0 < 1; idx_0++){
      uint64_t roffset_0 =  8000*idx_0 ;
      load_data(arg_0 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 8, 0, 0, 0);
      spadoffset_0 = spadoffset_0 + 8;
    } 
    /// %1 = ADORA.BlockLoad %arg0 [%arg1, 0] : memref<2000x2000xf32> -> memref<1x2000xf32>  {Id = "1", KernelName = "cholesky_kernel_1"}
    uint64_t dramoffset_1 = 8000 * arg_1;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_0 + dramoffset_1 + roffset_1, 0x18000 + spadoffset_1, 8000, 0, 0, 0);
    spadoffset_1 = spadoffset_1 + 8000;
    
    int int_2 = arg_1;
    int int_3 = arg_1;
    int int_4 = arg_1;
    int int_5 = arg_1;
    volatile unsigned short cin[25][3] __attribute__((aligned(8))) = {
    		{0x0000, 0xf800, 0x0028},
    		{0x0013, 0x0000, 0x0029},
    		{0x0000, 0x0100, 0x002a},
    		{0x0000, 0x0000, 0x002b},
    		{0x0800, 0xf800, 0x0030},
    		{0x0013, 0x0000, 0x0031},
    		{0x0000, 0x8f00, 0x0032},
    		{0x0000, 0x0000, 0x0033},
    		{0x0000, 0x0010, 0x0070},
    		{0x0030, 0x0000, 0x0078},
    		{0xc000, 0x0000, 0x0100},
    		{0x0000, 0x0400, 0x0108},
    		{0x060e, 0x0018, 0x0149},
    		{0x0000, 0x0000, 0x0198},
    		{0x0010, 0x0008, 0x01e1},
    		{0x0000, 0x0400, 0x01e2},
    		{0xe140, 0x014f, 0x01e3},
    		{0x0000, 0x0000, 0x01e4},
    		{0x0000, 0x0000, 0x0230},
    		{0x000d, 0x0008, 0x0279},
    		{0x0000, 0x0000, 0x02c8},
    		{0x2000, 0xf800, 0x0310},
    		{0x0013, 0x0000, 0x0311},
    		{0x0000, 0x0100, 0x0312},
    		{0x0000, 0x0000, 0x0313},
    	};
    
    cin[21][1] = (int_2 << 0xa) | (cin[21][1] & 0x3ff);
    cin[22][0] = (int_2 >> 0x6) | (cin[22][0] & 0xfc00);
    cin[0][1] = (int_3 << 0xa) | (cin[0][1] & 0x3ff);
    cin[1][0] = (int_3 >> 0x6) | (cin[1][0] & 0xfc00);
    cin[4][1] = (int_4 << 0xa) | (cin[4][1] & 0x3ff);
    cin[5][0] = (int_4 >> 0x6) | (cin[5][0] & 0xfc00);
    cin[16][0] = (int_5 << 0xc) | (cin[16][0] & 0xfff);
    cin[16][1] = (int_5 >> 0x8) | (cin[16][1] & 0xff00);
    load_cfg((void*)cin, 0x20000, 150, 0, 0);
    config(0x0, 25, 0, 0);
    execute(0x8030, 0, 0);
    /// ADORA.BlockStore %2, %arg0 [%arg1, %arg1] : memref<1x2xf32> -> memref<2000x2000xf32>  {Id = "2", KernelName = "cholesky_kernel_1"}
    uint64_t dramoffset_2 = 8 * arg_1 + 4 * arg_1;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_0 + dramoffset_2 + roffset_2, 0xa000 + spadoffset_2, 8, 0, 0);
    spadoffset_2 = spadoffset_2 + 8;
    
}
