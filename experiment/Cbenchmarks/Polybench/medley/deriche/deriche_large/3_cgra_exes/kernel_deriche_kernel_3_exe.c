
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void deriche_kernel_3(void* arg_0 ,void* arg_1 ,void* arg_2 ,void* arg_3 ,void* arg_4){
    for (int int_5 = 0; int_5 < 4096; int_5 = int_5 + 1){
      for (int int_6 = 0; int_6 < 2160; int_6 = int_6 + 1080){
        /// %0 = ADORA.BlockLoad %arg0 [] : memref<f32> -> memref<2xf32>  {Id = "0", KernelName = ""}
        load_data(arg_0, 0x10000, 8, 0, 0, 0);
        /// %1 = ADORA.BlockLoad %arg2 [] : memref<f32> -> memref<2xf32>  {Id = "1", KernelName = ""}
        load_data(arg_2, 0x0, 8, 0, 0, 0);
        /// %2 = ADORA.BlockLoad %arg1 [] : memref<f32> -> memref<2xf32>  {Id = "3", KernelName = ""}
        load_data(arg_1, 0x2000, 8, 0, 0, 0);
        /// %3 = ADORA.BlockLoad %arg3 [%arg5, %arg6] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "6", KernelName = ""}
        uint64_t dramoffset_6 = 8640 * int_5 + 4 * int_6;
        uint64_t spadoffset_6 = 0;
        for(int idx_0 = 0; idx_0 < 1; idx_0++){
          uint64_t roffset_6 =  8640*idx_0 ;
          load_data(arg_3 + dramoffset_6 + roffset_6, 0x8000 + spadoffset_6, 4320, 0, 0, 0);
          spadoffset_6 = spadoffset_6 + 4320;
        } 
        /// %4 = ADORA.BlockLoad %arg3 [%arg5, %arg6] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "7", KernelName = ""}
        uint64_t dramoffset_7 = 8640 * int_5 + 4 * int_6;
        uint64_t spadoffset_7 = 0;
        for(int idx_0 = 0; idx_0 < 1; idx_0++){
          uint64_t roffset_7 =  8640*idx_0 ;
          load_data(arg_3 + dramoffset_7 + roffset_7, 0x4000 + spadoffset_7, 4320, 0, 0, 0);
          spadoffset_7 = spadoffset_7 + 4320;
        } 
        /// %5 = ADORA.BlockLoad %arg4 [%arg5, %arg6] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "8", KernelName = ""}
        uint64_t dramoffset_8 = 8640 * int_5 + 4 * int_6;
        uint64_t spadoffset_8 = 0;
        for(int idx_0 = 0; idx_0 < 1; idx_0++){
          uint64_t roffset_8 =  8640*idx_0 ;
          load_data(arg_4 + dramoffset_8 + roffset_8, 0xa000 + spadoffset_8, 4320, 0, 0, 0);
          spadoffset_8 = spadoffset_8 + 4320;
        } 
        volatile unsigned short cin[74][3] __attribute__((aligned(8))) = {
        		{0x3000, 0xe000, 0x0008},
        		{0x0010, 0x0000, 0x0009},
        		{0x0000, 0x0100, 0x000a},
        		{0x0000, 0x0000, 0x000b},
        		{0x3800, 0xe000, 0x0010},
        		{0x0010, 0x0000, 0x0011},
        		{0x0000, 0x9700, 0x0012},
        		{0x0200, 0x0000, 0x0013},
        		{0x0000, 0xe000, 0x0018},
        		{0x0010, 0x0000, 0x0019},
        		{0x0000, 0x0100, 0x001a},
        		{0x0000, 0x0000, 0x001b},
        		{0x0800, 0xe000, 0x0020},
        		{0x0010, 0x0000, 0x0021},
        		{0x0000, 0x0100, 0x0022},
        		{0x0000, 0x0000, 0x0023},
        		{0x1000, 0xe000, 0x0030},
        		{0x0010, 0x0000, 0x0031},
        		{0x0000, 0x8700, 0x0032},
        		{0x0000, 0x0000, 0x0033},
        		{0x2800, 0xe000, 0x0038},
        		{0x0010, 0x0000, 0x0039},
        		{0x0000, 0x0100, 0x003a},
        		{0x0000, 0x0000, 0x003b},
        		{0x2000, 0xe000, 0x0040},
        		{0x0010, 0x0000, 0x0041},
        		{0x0000, 0x0100, 0x0042},
        		{0x0000, 0x0000, 0x0043},
        		{0x0000, 0x0000, 0x0058},
        		{0x0003, 0x0000, 0x0060},
        		{0x8000, 0x0000, 0x0068},
        		{0x0000, 0x0031, 0x0070},
        		{0x0020, 0x0002, 0x0078},
        		{0x0000, 0x0000, 0x0080},
        		{0x0000, 0x0000, 0x0088},
        		{0x4598, 0xbf1b, 0x00a0},
        		{0x000d, 0x0004, 0x00a1},
        		{0x060e, 0x0026, 0x00a9},
        		{0x44fd, 0x3f57, 0x00b8},
        		{0x000d, 0x0002, 0x00b9},
        		{0x35c4, 0xbe41, 0x00c8},
        		{0x000d, 0x0004, 0x00c9},
        		{0x0003, 0x0000, 0x00e9},
        		{0x4040, 0x000c, 0x00f0},
        		{0x0000, 0x0180, 0x00f8},
        		{0x0000, 0x0180, 0x0100},
        		{0x0003, 0x0000, 0x0101},
        		{0x0000, 0x0180, 0x0108},
        		{0x0000, 0x0000, 0x0110},
        		{0x040e, 0x0034, 0x0131},
        		{0x008e, 0x0012, 0x0139},
        		{0x0200, 0x0000, 0x0178},
        		{0x0003, 0x0000, 0x0179},
        		{0x0003, 0x0000, 0x0191},
        		{0x0000, 0x0400, 0x0208},
        		{0x0003, 0x0000, 0x0209},
        		{0x0003, 0x0000, 0x0221},
        		{0xb54c, 0x3de1, 0x0248},
        		{0x000d, 0x0040, 0x0249},
        		{0x0002, 0x0030, 0x0298},
        		{0x0200, 0x0000, 0x02a0},
        		{0x2000, 0x0000, 0x02b0},
        		{0x0000, 0xe000, 0x02d8},
        		{0x0010, 0x0000, 0x02d9},
        		{0x0000, 0x0100, 0x02da},
        		{0x0000, 0x0000, 0x02db},
        		{0x1000, 0xe000, 0x02e8},
        		{0x0010, 0x0000, 0x02e9},
        		{0x0000, 0x8d00, 0x02ea},
        		{0x0000, 0x0000, 0x02eb},
        		{0x0800, 0xe000, 0x02f0},
        		{0x0010, 0x0000, 0x02f1},
        		{0x0000, 0x8d00, 0x02f2},
        		{0x0200, 0x0000, 0x02f3},
        	};
        
        load_cfg((void*)cin, 0x20000, 444, 0, 0);
        config(0x0, 74, 0, 0);
        execute(0xdef, 0, 0);
        /// ADORA.BlockStore %6, %arg4 [%arg5, %arg6] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "9", KernelName = ""}
        uint64_t dramoffset_9 = 4320 * int_5 + 4 * int_6;
        uint64_t spadoffset_9 = 0;
        uint64_t roffset_9 = 0;
        store(arg_4 + dramoffset_9 + roffset_9, 0x6000 + spadoffset_9, 4320, 0, 0);
        spadoffset_9 = spadoffset_9 + 4320;
        
        /// ADORA.BlockStore %9, %arg1 [] : memref<2xf32> -> memref<f32>  {Id = "13", KernelName = ""}
        store(arg_1, 0x12000, 8, 0, 0);
        /// ADORA.BlockStore %8, %arg2 [] : memref<2xf32> -> memref<f32>  {Id = "11", KernelName = ""}
        store(arg_2, 0xc000, 8, 0, 0);
        /// ADORA.BlockStore %7, %arg0 [] : memref<2xf32> -> memref<f32>  {Id = "10", KernelName = ""}
        store(arg_0, 0x14000, 8, 0, 0);
        }
        }
}
