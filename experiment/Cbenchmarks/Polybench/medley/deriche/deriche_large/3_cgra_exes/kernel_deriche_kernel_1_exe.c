
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void deriche_kernel_1(void* arg_0 ,void* arg_1 ,void* arg_2 ,void* arg_3 ,void* arg_4 ,void* arg_5){
    for (int int_6 = 0; int_6 < 4096; int_6 = int_6 + 1){
      for (int int_7 = 0; int_7 < 2160; int_7 = int_7 + 1080){
        /// %0 = ADORA.BlockLoad %arg2 [] : memref<f32> -> memref<2xf32>  {Id = "0", KernelName = ""}
        load_data(arg_2, 0x0, 8, 0, 0, 0);
        /// %1 = ADORA.BlockLoad %arg0 [] : memref<f32> -> memref<2xf32>  {Id = "1", KernelName = ""}
        load_data(arg_0, 0x2000, 8, 0, 0, 0);
        /// %2 = ADORA.BlockLoad %arg3 [] : memref<f32> -> memref<2xf32>  {Id = "3", KernelName = ""}
        load_data(arg_3, 0x4000, 8, 0, 0, 0);
        /// %3 = ADORA.BlockLoad %arg1 [] : memref<f32> -> memref<2xf32>  {Id = "4", KernelName = ""}
        load_data(arg_1, 0x10000, 8, 0, 0, 0);
        int int_8 = int_7 * -1;
        int int_9 = int_8 + 1080;
        /// %6 = ADORA.BlockLoad %arg5 [%arg6, %5] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "8", KernelName = ""}
        uint64_t dramoffset_8 = 8640 * int_6 + 4 * int_9;
        uint64_t spadoffset_8 = 0;
        for(int idx_0 = 0; idx_0 < 1; idx_0++){
          uint64_t roffset_8 =  8640*idx_0 ;
          load_data(arg_5 + dramoffset_8 + roffset_8, 0x8000 + spadoffset_8, 4320, 0, 0, 0);
          spadoffset_8 = spadoffset_8 + 4320;
        } 
        volatile unsigned short cin[69][3] __attribute__((aligned(8))) = {
        		{0x1800, 0xe000, 0x0008},
        		{0x0010, 0x0000, 0x0009},
        		{0x0000, 0x8b00, 0x000a},
        		{0x0200, 0x0000, 0x000b},
        		{0x0000, 0xe000, 0x0010},
        		{0x0010, 0x0000, 0x0011},
        		{0x0000, 0x0100, 0x0012},
        		{0x0000, 0x0000, 0x0013},
        		{0x1000, 0xe000, 0x0018},
        		{0x0010, 0x0000, 0x0019},
        		{0x0000, 0x0100, 0x001a},
        		{0x0000, 0x0000, 0x001b},
        		{0x0800, 0xe000, 0x0020},
        		{0x0010, 0x0000, 0x0021},
        		{0x0000, 0x0100, 0x0022},
        		{0x0000, 0x0000, 0x0023},
        		{0x0800, 0xe000, 0x0028},
        		{0x0010, 0x0000, 0x0029},
        		{0x0000, 0x8700, 0x002a},
        		{0x0000, 0x0000, 0x002b},
        		{0xe000, 0xe3ff, 0x0030},
        		{0x0010, 0x0000, 0x0031},
        		{0x0000, 0x0100, 0x0032},
        		{0x0000, 0x0000, 0x0033},
        		{0x0002, 0x0000, 0x0058},
        		{0x0200, 0x0002, 0x0060},
        		{0x0200, 0x0002, 0x0068},
        		{0x0000, 0x0002, 0x0070},
        		{0x0000, 0x0000, 0x0078},
        		{0x1714, 0xbe3c, 0x0098},
        		{0x000d, 0x0004, 0x0099},
        		{0x6028, 0x3dea, 0x00a8},
        		{0x000d, 0x0010, 0x00a9},
        		{0x44fd, 0x3f57, 0x00b0},
        		{0x000d, 0x0010, 0x00b1},
        		{0x0000, 0x0000, 0x00e8},
        		{0x0000, 0x0000, 0x00f0},
        		{0x0003, 0x0000, 0x00f1},
        		{0x0000, 0x0000, 0x00f8},
        		{0x000e, 0x0022, 0x0131},
        		{0x020e, 0x0026, 0x0139},
        		{0x0000, 0x0000, 0x0180},
        		{0x0003, 0x0000, 0x0181},
        		{0x040e, 0x0044, 0x01c1},
        		{0x0001, 0x0000, 0x0209},
        		{0x0000, 0x0000, 0x0210},
        		{0x0003, 0x0000, 0x0211},
        		{0x4598, 0xbf1b, 0x0258},
        		{0x000d, 0x0040, 0x0259},
        		{0x2300, 0x0000, 0x0298},
        		{0x0000, 0x0030, 0x02a0},
        		{0x0002, 0x0020, 0x02a8},
        		{0x0200, 0x0000, 0x02b0},
        		{0xe800, 0xe3ff, 0x02d8},
        		{0x0010, 0x0000, 0x02d9},
        		{0x0000, 0x9100, 0x02da},
        		{0x0200, 0x0000, 0x02db},
        		{0x1000, 0xe000, 0x02e0},
        		{0x0010, 0x0000, 0x02e1},
        		{0x0000, 0x9100, 0x02e2},
        		{0x0000, 0x0000, 0x02e3},
        		{0x0000, 0xe000, 0x02e8},
        		{0x0010, 0x0000, 0x02e9},
        		{0x0000, 0x0100, 0x02ea},
        		{0x0000, 0x0000, 0x02eb},
        		{0x0000, 0xe000, 0x02f8},
        		{0x0010, 0x0000, 0x02f9},
        		{0x0000, 0x8d00, 0x02fa},
        		{0x0000, 0x0000, 0x02fb},
        	};
        
        load_cfg((void*)cin, 0x20000, 414, 0, 0);
        config(0x0, 69, 0, 0);
        execute(0x173f, 0, 0);
        int int_10 = int_7 * -1;
        int int_11 = int_10 + 1080;
        /// ADORA.BlockStore %7, %arg4 [%arg6, %13] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "9", KernelName = ""}
        uint64_t dramoffset_9 = 4320 * int_6 + 4 * int_11;
        uint64_t spadoffset_9 = 0;
        uint64_t roffset_9 = 0;
        store(arg_4 + dramoffset_9 + roffset_9, 0x12000 + spadoffset_9, 4320, 0, 0);
        spadoffset_9 = spadoffset_9 + 4320;
        
        /// ADORA.BlockStore %11, %arg1 [] : memref<2xf32> -> memref<f32>  {Id = "14", KernelName = ""}
        store(arg_1, 0xa000, 8, 0, 0);
        /// ADORA.BlockStore %10, %arg3 [] : memref<2xf32> -> memref<f32>  {Id = "13", KernelName = ""}
        store(arg_3, 0x18000, 8, 0, 0);
        /// ADORA.BlockStore %9, %arg0 [] : memref<2xf32> -> memref<f32>  {Id = "11", KernelName = ""}
        store(arg_0, 0x14000, 8, 0, 0);
        /// ADORA.BlockStore %8, %arg2 [] : memref<2xf32> -> memref<f32>  {Id = "10", KernelName = ""}
        store(arg_2, 0x6000, 8, 0, 0);
        }
        }
}
