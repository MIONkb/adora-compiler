
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void deriche_kernel_0(void* arg_0 ,void* arg_1 ,void* arg_2 ,void* arg_3 ,void* arg_4){
    for (int int_5 = 0; int_5 < 4096; int_5 = int_5 + 1){
      for (int int_6 = 0; int_6 < 2160; int_6 = int_6 + 1080){
        /// %0 = ADORA.BlockLoad %arg2 [] : memref<f32> -> memref<2xf32>  {Id = "0", KernelName = ""}
        load_data(arg_2, 0x10000, 8, 0, 0, 0);
        /// %1 = ADORA.BlockLoad %arg1 [] : memref<f32> -> memref<2xf32>  {Id = "1", KernelName = ""}
        load_data(arg_1, 0x12000, 8, 0, 0, 0);
        /// %2 = ADORA.BlockLoad %arg0 [] : memref<f32> -> memref<2xf32>  {Id = "3", KernelName = ""}
        load_data(arg_0, 0x14000, 8, 0, 0, 0);
        /// %3 = ADORA.BlockLoad %arg3 [%arg5, %arg6] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "6", KernelName = ""}
        uint64_t dramoffset_6 = 8640 * int_5 + 4 * int_6;
        uint64_t spadoffset_6 = 0;
        for(int idx_0 = 0; idx_0 < 1; idx_0++){
          uint64_t roffset_6 =  8640*idx_0 ;
          load_data(arg_3 + dramoffset_6 + roffset_6, 0x18000 + spadoffset_6, 4320, 0, 0, 0);
          spadoffset_6 = spadoffset_6 + 4320;
        } 
        volatile unsigned short cin[62][3] __attribute__((aligned(8))) = {
        		{0x0800, 0xe000, 0x0008},
        		{0x0010, 0x0000, 0x0009},
        		{0x0000, 0x8b00, 0x000a},
        		{0x0200, 0x0000, 0x000b},
        		{0x0000, 0xe000, 0x0010},
        		{0x0010, 0x0000, 0x0011},
        		{0x0000, 0x9500, 0x0012},
        		{0x0000, 0x0000, 0x0013},
        		{0x0032, 0x0000, 0x0058},
        		{0x0000, 0x0003, 0x0060},
        		{0x0000, 0x1000, 0x00e8},
        		{0x0000, 0x1000, 0x00f0},
        		{0x0000, 0x0c00, 0x0178},
        		{0x0000, 0x1080, 0x0180},
        		{0x060e, 0x0038, 0x01c9},
        		{0x0000, 0x2000, 0x0208},
        		{0x0100, 0x3000, 0x0210},
        		{0x8000, 0x2000, 0x0218},
        		{0x0000, 0x0000, 0x0219},
        		{0x8000, 0x0004, 0x0220},
        		{0x0000, 0x000c, 0x0228},
        		{0x0000, 0x0080, 0x0230},
        		{0x4598, 0xbf1b, 0x0248},
        		{0x000d, 0x0040, 0x0249},
        		{0x44fd, 0x3f57, 0x0250},
        		{0x000d, 0x0008, 0x0251},
        		{0xb54c, 0x3de1, 0x0258},
        		{0x000d, 0x0006, 0x0259},
        		{0x020e, 0x0014, 0x0261},
        		{0x020e, 0x0014, 0x0269},
        		{0x35c4, 0xbe41, 0x0278},
        		{0x000d, 0x0008, 0x0279},
        		{0x0002, 0x0000, 0x0298},
        		{0x0010, 0x0000, 0x02a0},
        		{0x0300, 0x0000, 0x02a8},
        		{0x3000, 0x0000, 0x02b8},
        		{0x0000, 0x0003, 0x02c0},
        		{0x0000, 0x0001, 0x02c8},
        		{0x0800, 0xe000, 0x02d8},
        		{0x0010, 0x0000, 0x02d9},
        		{0x0000, 0x0100, 0x02da},
        		{0x0000, 0x0000, 0x02db},
        		{0x0000, 0xe000, 0x02e0},
        		{0x0010, 0x0000, 0x02e1},
        		{0x0000, 0x0100, 0x02e2},
        		{0x0000, 0x0000, 0x02e3},
        		{0x1000, 0xe000, 0x02e8},
        		{0x0010, 0x0000, 0x02e9},
        		{0x0000, 0x0100, 0x02ea},
        		{0x0000, 0x0000, 0x02eb},
        		{0x3800, 0xe000, 0x02f0},
        		{0x0010, 0x0000, 0x02f1},
        		{0x0000, 0x9300, 0x02f2},
        		{0x0000, 0x0000, 0x02f3},
        		{0x0800, 0xe000, 0x02f8},
        		{0x0010, 0x0000, 0x02f9},
        		{0x0000, 0x8900, 0x02fa},
        		{0x0200, 0x0000, 0x02fb},
        		{0x2000, 0xe000, 0x0310},
        		{0x0010, 0x0000, 0x0311},
        		{0x0000, 0x0100, 0x0312},
        		{0x0000, 0x0000, 0x0313},
        	};
        
        load_cfg((void*)cin, 0x20000, 372, 0, 0);
        config(0x0, 62, 0, 0);
        execute(0x9f03, 0, 0);
        /// ADORA.BlockStore %4, %arg4 [%arg5, %arg6] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "7", KernelName = ""}
        uint64_t dramoffset_7 = 4320 * int_5 + 4 * int_6;
        uint64_t spadoffset_7 = 0;
        uint64_t roffset_7 = 0;
        store(arg_4 + dramoffset_7 + roffset_7, 0x16000 + spadoffset_7, 4320, 0, 0);
        spadoffset_7 = spadoffset_7 + 4320;
        
        /// ADORA.BlockStore %7, %arg0 [] : memref<2xf32> -> memref<f32>  {Id = "13", KernelName = ""}
        store(arg_0, 0x0, 8, 0, 0);
        /// ADORA.BlockStore %6, %arg1 [] : memref<2xf32> -> memref<f32>  {Id = "11", KernelName = ""}
        store(arg_1, 0x2000, 8, 0, 0);
        /// ADORA.BlockStore %5, %arg2 [] : memref<2xf32> -> memref<f32>  {Id = "10", KernelName = ""}
        store(arg_2, 0x1a000, 8, 0, 0);
        }
        }
}
