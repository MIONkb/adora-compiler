
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void cholesky_kernel_0(void* arg_0 ,int arg_1 ,int arg_2){
    /// %0 = ADORA.BlockLoad %arg0 [%arg1, 0] : memref<2000x2000xf32> -> memref<1x2000xf32>  {Id = "0", KernelName = "cholesky_kernel_0"}
    uint64_t dramoffset_0 = 8000 * arg_1;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x10000 + spadoffset_0, 8000, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 8000;
    
    /// %1 = ADORA.BlockLoad %arg0 [%arg2, 0] : memref<2000x2000xf32> -> memref<1x2000xf32>  {Id = "1", KernelName = "cholesky_kernel_0"}
    uint64_t dramoffset_1 = 8000 * arg_2;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_0 + dramoffset_1 + roffset_1, 0x0 + spadoffset_1, 8000, 0, 0, 0);
    spadoffset_1 = spadoffset_1 + 8000;
    
    /// %2 = ADORA.BlockLoad %arg0 [%arg1, %arg2] : memref<2000x2000xf32> -> memref<1x2xf32>  {Id = "2", KernelName = "cholesky_kernel_0"}
    uint64_t dramoffset_2 = 8000 * arg_1 + 4 * arg_2;
    uint64_t spadoffset_2 = 0;
    for(int idx_0 = 0; idx_0 < 1; idx_0++){
      uint64_t roffset_2 =  8000*idx_0 ;
      load_data(arg_0 + dramoffset_2 + roffset_2, 0x12000 + spadoffset_2, 8, 0, 0, 0);
      spadoffset_2 = spadoffset_2 + 8;
    } 
    int int_3 = arg_2;
    int int_4 = arg_2;
    int int_5 = arg_2;
    int int_6 = arg_2;
    int int_7 = arg_2;
    volatile unsigned short cin[32][3] __attribute__((aligned(8))) = {
    		{0x2000, 0xf800, 0x0010},
    		{0x0013, 0x0000, 0x0011},
    		{0x0000, 0x0100, 0x0012},
    		{0x0000, 0x0000, 0x0013},
    		{0x0000, 0x0010, 0x0058},
    		{0x0010, 0x0006, 0x00a9},
    		{0x0000, 0x0400, 0x00aa},
    		{0xe180, 0x014f, 0x00ab},
    		{0x0000, 0x0000, 0x00ac},
    		{0xc000, 0x0000, 0x00e8},
    		{0x0040, 0x0000, 0x00f0},
    		{0x0001, 0x0000, 0x00f1},
    		{0x000d, 0x0016, 0x0131},
    		{0x00c0, 0x0000, 0x0178},
    		{0x0003, 0x0000, 0x0181},
    		{0x0000, 0x1000, 0x0208},
    		{0x0000, 0x0008, 0x0210},
    		{0x0a0e, 0x0034, 0x0251},
    		{0x0000, 0x0000, 0x0298},
    		{0x0000, 0x0000, 0x02a0},
    		{0x1000, 0xf800, 0x02d8},
    		{0x0013, 0x0000, 0x02d9},
    		{0x0000, 0x0100, 0x02da},
    		{0x0000, 0x0000, 0x02db},
    		{0x2000, 0xf800, 0x02e0},
    		{0x0013, 0x0000, 0x02e1},
    		{0x0000, 0x0100, 0x02e2},
    		{0x0000, 0x0000, 0x02e3},
    		{0x0000, 0xf800, 0x02e8},
    		{0x0013, 0x0000, 0x02e9},
    		{0x0000, 0x9300, 0x02ea},
    		{0x0000, 0x0000, 0x02eb},
    	};
    
    cin[24][1] = (int_3 << 0xa) | (cin[24][1] & 0x3ff);
    cin[25][0] = (int_3 >> 0x6) | (cin[25][0] & 0xfc00);
    cin[0][1] = (int_4 << 0xa) | (cin[0][1] & 0x3ff);
    cin[1][0] = (int_4 >> 0x6) | (cin[1][0] & 0xfc00);
    cin[20][1] = (int_5 << 0xa) | (cin[20][1] & 0x3ff);
    cin[21][0] = (int_5 >> 0x6) | (cin[21][0] & 0xfc00);
    cin[28][1] = (int_6 << 0xa) | (cin[28][1] & 0x3ff);
    cin[29][0] = (int_6 >> 0x6) | (cin[29][0] & 0xfc00);
    cin[7][0] = (int_7 << 0xc) | (cin[7][0] & 0xfff);
    cin[7][1] = (int_7 >> 0x8) | (cin[7][1] & 0xff00);
    load_cfg((void*)cin, 0x20000, 192, 0, 0);
    config(0x0, 32, 0, 0);
    execute(0x302, 0, 0);
    /// ADORA.BlockStore %3, %arg0 [%arg1, %arg2] : memref<1x2xf32> -> memref<2000x2000xf32>  {Id = "2", KernelName = "cholesky_kernel_0"}
    uint64_t dramoffset_2 = 8 * arg_1 + 4 * arg_2;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_0 + dramoffset_2 + roffset_2, 0x14000 + spadoffset_2, 8, 0, 0);
    spadoffset_2 = spadoffset_2 + 8;
    
}
