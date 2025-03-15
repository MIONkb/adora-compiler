
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void cholesky_kernel_0(void* arg_0 ,int arg_1 ,int arg_2){
    /// %0 = ADORA.BlockLoad %arg0 [%arg1, 0] : memref<120x120xf32> -> memref<1x120xf32>  {Id = "0", KernelName = "cholesky_kernel_0"}
    uint64_t dramoffset_0 = 480 * arg_1;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 480, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 480;
    
    /// %1 = ADORA.BlockLoad %arg0 [%arg2, 0] : memref<120x120xf32> -> memref<1x120xf32>  {Id = "1", KernelName = "cholesky_kernel_0"}
    uint64_t dramoffset_1 = 480 * arg_2;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_0 + dramoffset_1 + roffset_1, 0x18000 + spadoffset_1, 480, 0, 0, 0);
    spadoffset_1 = spadoffset_1 + 480;
    
    /// %2 = ADORA.BlockLoad %arg0 [%arg1, %arg2] : memref<120x120xf32> -> memref<1x2xf32>  {Id = "2", KernelName = "cholesky_kernel_0"}
    uint64_t dramoffset_2 = 480 * arg_1 + 4 * arg_2;
    uint64_t spadoffset_2 = 0;
    for(int idx_0 = 0; idx_0 < 1; idx_0++){
      uint64_t roffset_2 =  480*idx_0 ;
      load_data(arg_0 + dramoffset_2 + roffset_2, 0x10000 + spadoffset_2, 8, 0, 0, 0);
      spadoffset_2 = spadoffset_2 + 8;
    } 
    int int_3 = arg_2;
    int int_4 = arg_2;
    int int_5 = arg_2;
    int int_6 = arg_2;
    int int_7 = arg_2;
    volatile unsigned short cin[31][3] __attribute__((aligned(8))) = {
    		{0x2000, 0xf800, 0x0020},
    		{0x0013, 0x0000, 0x0021},
    		{0x0000, 0x0100, 0x0022},
    		{0x0000, 0x0000, 0x0023},
    		{0x0000, 0x0000, 0x0070},
    		{0x0003, 0x0000, 0x0101},
    		{0x0003, 0x0000, 0x0191},
    		{0x0010, 0x0008, 0x01d1},
    		{0x0000, 0x0400, 0x01d2},
    		{0xe180, 0x014f, 0x01d3},
    		{0x0000, 0x0000, 0x01d4},
    		{0x0000, 0x000c, 0x0210},
    		{0x0000, 0x0000, 0x0218},
    		{0x2000, 0x0000, 0x0220},
    		{0x080e, 0x0034, 0x0251},
    		{0x020d, 0x0042, 0x0269},
    		{0x0000, 0x0000, 0x0298},
    		{0x0000, 0x0000, 0x02a0},
    		{0x0002, 0x0000, 0x02b8},
    		{0x0000, 0xf800, 0x02d8},
    		{0x0013, 0x0000, 0x02d9},
    		{0x0000, 0x0100, 0x02da},
    		{0x0000, 0x0000, 0x02db},
    		{0x0000, 0xf800, 0x02e8},
    		{0x0013, 0x0000, 0x02e9},
    		{0x0000, 0x9100, 0x02ea},
    		{0x0000, 0x0000, 0x02eb},
    		{0x2000, 0xf800, 0x02f8},
    		{0x0013, 0x0000, 0x02f9},
    		{0x0000, 0x0100, 0x02fa},
    		{0x0000, 0x0000, 0x02fb},
    	};
    
    cin[0][1] = (int_3 << 0xa) | (cin[0][1] & 0x3ff);
    cin[1][0] = (int_3 >> 0x6) | (cin[1][0] & 0xfc00);
    cin[27][1] = (int_4 << 0xa) | (cin[27][1] & 0x3ff);
    cin[28][0] = (int_4 >> 0x6) | (cin[28][0] & 0xfc00);
    cin[19][1] = (int_5 << 0xa) | (cin[19][1] & 0x3ff);
    cin[20][0] = (int_5 >> 0x6) | (cin[20][0] & 0xfc00);
    cin[23][1] = (int_6 << 0xa) | (cin[23][1] & 0x3ff);
    cin[24][0] = (int_6 >> 0x6) | (cin[24][0] & 0xfc00);
    cin[9][0] = (int_7 << 0xc) | (cin[9][0] & 0xfff);
    cin[9][1] = (int_7 >> 0x8) | (cin[9][1] & 0xff00);
    load_cfg((void*)cin, 0x20000, 186, 0, 0);
    config(0x0, 31, 0, 0);
    execute(0x1108, 0, 0);
    fence(1);

    /// ADORA.BlockStore %3, %arg0 [%arg1, %arg2] : memref<1x2xf32> -> memref<120x120xf32>  {Id = "3", KernelName = "cholesky_kernel_0"}
    uint64_t dramoffset_3 = 8 * arg_1 + 4 * arg_2;
    uint64_t spadoffset_3 = 0;
    uint64_t roffset_3 = 0;
    store(arg_0 + dramoffset_3 + roffset_3, 0x0 + spadoffset_3, 8, 0, 0);
    spadoffset_3 = spadoffset_3 + 8;
    
    fence(1);
}
