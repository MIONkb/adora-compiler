
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void cholesky_kernel_1(void* arg_0 ,int arg_1){
    /// %0 = ADORA.BlockLoad %arg0 [%arg1, %arg1] : memref<120x120xf32> -> memref<1x2xf32>  {Id = "0", KernelName = "cholesky_kernel_1"}
    uint64_t dramoffset_0 = 480 * arg_1 + 4 * arg_1;
    uint64_t spadoffset_0 = 0;
    for(int idx_0 = 0; idx_0 < 1; idx_0++){
      uint64_t roffset_0 =  480*idx_0 ;
      load_data(arg_0 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 8, 0, 0, 0);
      spadoffset_0 = spadoffset_0 + 8;
    } 
    /// %1 = ADORA.BlockLoad %arg0 [%arg1, 0] : memref<120x120xf32> -> memref<1x120xf32>  {Id = "1", KernelName = "cholesky_kernel_1"}
    uint64_t dramoffset_1 = 480 * arg_1;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_0 + dramoffset_1 + roffset_1, 0x18000 + spadoffset_1, 480, 0, 0, 0);
    spadoffset_1 = spadoffset_1 + 480;
    
    int int_2 = arg_1;
    int int_3 = arg_1;
    int int_4 = arg_1;
    int int_5 = arg_1;
    volatile unsigned short cin[27][3] __attribute__((aligned(8))) = {
    		{0x0000, 0xf800, 0x0020},
    		{0x0013, 0x0000, 0x0021},
    		{0x0000, 0x0100, 0x0022},
    		{0x0000, 0x0000, 0x0023},
    		{0x0000, 0xf800, 0x0028},
    		{0x0013, 0x0000, 0x0029},
    		{0x0000, 0x9300, 0x002a},
    		{0x0200, 0x0000, 0x002b},
    		{0x0000, 0x0000, 0x0070},
    		{0x0400, 0x0000, 0x0078},
    		{0x0a0e, 0x0018, 0x00c1},
    		{0x0001, 0x0000, 0x0110},
    		{0x0000, 0x0200, 0x0118},
    		{0x0000, 0x8000, 0x01a0},
    		{0x1000, 0x0000, 0x01a8},
    		{0x0010, 0x0002, 0x01f1},
    		{0x0000, 0x0400, 0x01f2},
    		{0xe180, 0x014f, 0x01f3},
    		{0x0000, 0x0000, 0x01f4},
    		{0x0000, 0x0c00, 0x0230},
    		{0x0000, 0x0080, 0x0238},
    		{0x000d, 0x0006, 0x0281},
    		{0x0010, 0x0000, 0x02c8},
    		{0x2000, 0xf800, 0x0308},
    		{0x0013, 0x0000, 0x0309},
    		{0x0000, 0x0100, 0x030a},
    		{0x0000, 0x0000, 0x030b},
    	};
    
    cin[23][1] = (int_2 << 0xa) | (cin[23][1] & 0x3ff);
    cin[24][0] = (int_2 >> 0x6) | (cin[24][0] & 0xfc00);
    cin[0][1] = (int_3 << 0xa) | (cin[0][1] & 0x3ff);
    cin[1][0] = (int_3 >> 0x6) | (cin[1][0] & 0xfc00);
    cin[4][1] = (int_4 << 0xa) | (cin[4][1] & 0x3ff);
    cin[5][0] = (int_4 >> 0x6) | (cin[5][0] & 0xfc00);
    cin[17][0] = (int_5 << 0xc) | (cin[17][0] & 0xfff);
    cin[17][1] = (int_5 >> 0x8) | (cin[17][1] & 0xff00);
    load_cfg((void*)cin, 0x20000, 162, 0, 0);
    config(0x0, 27, 0, 0);
    execute(0x4018, 0, 0);
    // fence(1);

    /// ADORA.BlockStore %2, %arg0 [%arg1, %arg1] : memref<1x2xf32> -> memref<120x120xf32>  {Id = "2", KernelName = "cholesky_kernel_1"}
    uint64_t dramoffset_2 = 8 * arg_1 + 4 * arg_1;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_0 + dramoffset_2 + roffset_2, 0x8000 + spadoffset_2, 8, 0, 0);
    spadoffset_2 = spadoffset_2 + 8;
    

    fence(1);
}
