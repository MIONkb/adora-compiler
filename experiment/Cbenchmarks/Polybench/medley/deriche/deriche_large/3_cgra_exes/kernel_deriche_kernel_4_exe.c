
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void deriche_kernel_4(void* arg_0 ,void* arg_1 ,void* arg_2 ,void* arg_3 ,void* arg_4 ,void* arg_5){
    for (int int_6 = 0; int_6 < 4096; int_6 = int_6 + 1){
      for (int int_7 = 0; int_7 < 2160; int_7 = int_7 + 1080){
        /// %0 = ADORA.BlockLoad %arg0 [] : memref<f32> -> memref<2xf32>  {Id = "0", KernelName = ""}
        load_data(arg_0, 0x0, 8, 0, 0, 0);
        /// %1 = ADORA.BlockLoad %arg2 [] : memref<f32> -> memref<2xf32>  {Id = "1", KernelName = ""}
        load_data(arg_2, 0x18000, 8, 0, 0, 0);
        /// %2 = ADORA.BlockLoad %arg1 [] : memref<f32> -> memref<2xf32>  {Id = "3", KernelName = ""}
        load_data(arg_1, 0x10000, 8, 0, 0, 0);
        /// %3 = ADORA.BlockLoad %arg3 [] : memref<f32> -> memref<2xf32>  {Id = "4", KernelName = ""}
        load_data(arg_3, 0x8000, 8, 0, 0, 0);
        int int_8 = int_6 * -1;
        int int_9 = int_8 + 4095;
        /// %6 = ADORA.BlockLoad %arg5 [%5, %arg7] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "5", KernelName = ""}
        uint64_t dramoffset_5 = 8640 * int_9 + 4 * int_7;
        uint64_t spadoffset_5 = 0;
        for(int idx_0 = 0; idx_0 < 1; idx_0++){
          uint64_t roffset_5 =  8640*idx_0 ;
          load_data(arg_5 + dramoffset_5 + roffset_5, 0x1a000 + spadoffset_5, 4320, 0, 0, 0);
          spadoffset_5 = spadoffset_5 + 4320;
        } 
        int int_10 = int_6 * -1;
        int int_11 = int_10 + 4095;
        /// %9 = ADORA.BlockLoad %arg4 [%8, %arg7] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "6", KernelName = ""}
        uint64_t dramoffset_6 = 8640 * int_11 + 4 * int_7;
        uint64_t spadoffset_6 = 0;
        for(int idx_0 = 0; idx_0 < 1; idx_0++){
          uint64_t roffset_6 =  8640*idx_0 ;
          load_data(arg_4 + dramoffset_6 + roffset_6, 0xa000 + spadoffset_6, 4320, 0, 0, 0);
          spadoffset_6 = spadoffset_6 + 4320;
        } 
        volatile unsigned short cin[78][3] __attribute__((aligned(8))) = {
        		{0x0000, 0xe000, 0x0010},
        		{0x0010, 0x0000, 0x0011},
        		{0x0000, 0x0100, 0x0012},
        		{0x0000, 0x0000, 0x0013},
        		{0x1000, 0xe000, 0x0018},
        		{0x0010, 0x0000, 0x0019},
        		{0x0000, 0x8d00, 0x001a},
        		{0x0200, 0x0000, 0x001b},
        		{0x0800, 0xe000, 0x0020},
        		{0x0010, 0x0000, 0x0021},
        		{0x0000, 0x8700, 0x0022},
        		{0x0000, 0x0000, 0x0023},
        		{0x0000, 0xe000, 0x0028},
        		{0x0010, 0x0000, 0x0029},
        		{0x0000, 0x0100, 0x002a},
        		{0x0000, 0x0000, 0x002b},
        		{0x1000, 0xe000, 0x0030},
        		{0x0010, 0x0000, 0x0031},
        		{0x0000, 0x8900, 0x0032},
        		{0x0200, 0x0000, 0x0033},
        		{0x2800, 0xe000, 0x0040},
        		{0x0010, 0x0000, 0x0041},
        		{0x0000, 0x0100, 0x0042},
        		{0x0000, 0x0000, 0x0043},
        		{0x0200, 0x0000, 0x0058},
        		{0x0000, 0x0000, 0x0060},
        		{0x0022, 0x0000, 0x0068},
        		{0x0200, 0x0002, 0x0070},
        		{0x0000, 0x0002, 0x0078},
        		{0x0002, 0x0003, 0x0080},
        		{0x0000, 0x0000, 0x0088},
        		{0x1714, 0xbe3c, 0x00a0},
        		{0x000d, 0x0010, 0x00a1},
        		{0x4598, 0xbf1b, 0x00b8},
        		{0x000d, 0x0010, 0x00b9},
        		{0x0001, 0x0000, 0x00e9},
        		{0x0001, 0x0000, 0x0101},
        		{0x0000, 0x1000, 0x0110},
        		{0x2000, 0x0000, 0x0178},
        		{0x2000, 0x0000, 0x0190},
        		{0x0000, 0x1000, 0x01a0},
        		{0x000e, 0x0042, 0x01c1},
        		{0x004e, 0x0042, 0x01d9},
        		{0x0000, 0x0000, 0x0210},
        		{0x0000, 0x4000, 0x0218},
        		{0x1000, 0x0000, 0x0220},
        		{0x0004, 0x0030, 0x0228},
        		{0x0000, 0x0000, 0x0229},
        		{0x0000, 0x1080, 0x0230},
        		{0x6028, 0x3dea, 0x0258},
        		{0x000d, 0x0006, 0x0259},
        		{0x020e, 0x0022, 0x0269},
        		{0x44fd, 0x3f57, 0x0278},
        		{0x000d, 0x0008, 0x0279},
        		{0x0010, 0x0000, 0x02a0},
        		{0x3300, 0x0000, 0x02b8},
        		{0x0000, 0x0003, 0x02c0},
        		{0x0000, 0x0001, 0x02c8},
        		{0x0000, 0xe000, 0x02e0},
        		{0x0010, 0x0000, 0x02e1},
        		{0x0000, 0x0100, 0x02e2},
        		{0x0000, 0x0000, 0x02e3},
        		{0x1800, 0xe000, 0x02f8},
        		{0x0010, 0x0000, 0x02f9},
        		{0x0000, 0x8900, 0x02fa},
        		{0x0200, 0x0000, 0x02fb},
        		{0x3000, 0xe000, 0x0300},
        		{0x0010, 0x0000, 0x0301},
        		{0x0000, 0x9300, 0x0302},
        		{0x0000, 0x0000, 0x0303},
        		{0x2800, 0xe000, 0x0308},
        		{0x0010, 0x0000, 0x0309},
        		{0x0000, 0x0100, 0x030a},
        		{0x0000, 0x0000, 0x030b},
        		{0x0000, 0xe000, 0x0310},
        		{0x0010, 0x0000, 0x0311},
        		{0x0000, 0x0100, 0x0312},
        		{0x0000, 0x0000, 0x0313},
        	};
        
        load_cfg((void*)cin, 0x20000, 468, 0, 0);
        config(0x0, 78, 0, 0);
        execute(0xf2be, 0, 0);
        int int_12 = int_6 * -1;
        int int_13 = int_12 + 4095;
        /// ADORA.BlockStore %10, %arg4 [%16, %arg7] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "7", KernelName = ""}
        uint64_t dramoffset_7 = 4320 * int_13 + 4 * int_7;
        uint64_t spadoffset_7 = 0;
        uint64_t roffset_7 = 0;
        store(arg_4 + dramoffset_7 + roffset_7, 0x1c000 + spadoffset_7, 4320, 0, 0);
        spadoffset_7 = spadoffset_7 + 4320;
        
        /// ADORA.BlockStore %14, %arg3 [] : memref<2xf32> -> memref<f32>  {Id = "14", KernelName = ""}
        store(arg_3, 0x1e000, 8, 0, 0);
        /// ADORA.BlockStore %13, %arg1 [] : memref<2xf32> -> memref<f32>  {Id = "13", KernelName = ""}
        store(arg_1, 0x2000, 8, 0, 0);
        /// ADORA.BlockStore %12, %arg2 [] : memref<2xf32> -> memref<f32>  {Id = "11", KernelName = ""}
        store(arg_2, 0xc000, 8, 0, 0);
        /// ADORA.BlockStore %11, %arg0 [] : memref<2xf32> -> memref<f32>  {Id = "10", KernelName = ""}
        store(arg_0, 0x4000, 8, 0, 0);
        }
        }
}
