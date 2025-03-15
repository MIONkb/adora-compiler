
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void forward_kernel_0(void* arg_0 ,void* arg_1){
    for (int int_2 = 0; int_2 < 1; int_2 = int_2 + 1){
      /// %0 = ADORA.BlockLoad %arg0 [0, %c0, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_0 = 0;
      uint64_t spadoffset_0 = 0;
      uint64_t roffset_0 = 0;
      load_data(arg_0 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 8192, 0, 0, 0);
      spadoffset_0 = spadoffset_0 + 8192;
      
      /// %2 = ADORA.BlockLoad %arg0 [0, %c32, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "2", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_2 = 256 * 32;
      uint64_t spadoffset_2 = 0;
      uint64_t roffset_2 = 0;
      load_data(arg_0 + dramoffset_2 + roffset_2, 0x8000 + spadoffset_2, 8192, 0, 0, 0);
      spadoffset_2 = spadoffset_2 + 8192;
      
      /// %4 = ADORA.BlockLoad %arg0 [0, %c64, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "4", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_4 = 256 * 64;
      uint64_t spadoffset_4 = 0;
      uint64_t roffset_4 = 0;
      load_data(arg_0 + dramoffset_4 + roffset_4, 0x0 + spadoffset_4, 8192, 0, 0, 0);
      spadoffset_4 = spadoffset_4 + 8192;
      
      /// %6 = ADORA.BlockLoad %arg0 [0, %c96, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "6", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_6 = 256 * 96;
      uint64_t spadoffset_6 = 0;
      uint64_t roffset_6 = 0;
      load_data(arg_0 + dramoffset_6 + roffset_6, 0x10000 + spadoffset_6, 8192, 0, 0, 0);
      spadoffset_6 = spadoffset_6 + 8192;
      
      volatile unsigned short cin[63][3] __attribute__((aligned(8))) = {
      		{0x2000, 0x0000, 0x0008},
      		{0x0041, 0x0100, 0x0009},
      		{0x0000, 0x0100, 0x000a},
      		{0x0000, 0x0000, 0x000b},
      		{0x2800, 0x0000, 0x0010},
      		{0x0041, 0x0100, 0x0011},
      		{0x0000, 0x8b00, 0x0012},
      		{0x0200, 0x0000, 0x0013},
      		{0x2800, 0x0000, 0x0030},
      		{0x0041, 0x0100, 0x0031},
      		{0x0000, 0x8d00, 0x0032},
      		{0x0200, 0x0000, 0x0033},
      		{0x2000, 0x0000, 0x0040},
      		{0x0041, 0x0100, 0x0041},
      		{0x0000, 0x0100, 0x0042},
      		{0x0000, 0x0000, 0x0043},
      		{0x0000, 0x0000, 0x0058},
      		{0x0001, 0x0000, 0x0060},
      		{0x0000, 0x0000, 0x0080},
      		{0x0200, 0x0000, 0x0088},
      		{0x0000, 0x0000, 0x0098},
      		{0x0012, 0x0010, 0x0099},
      		{0x0000, 0x0000, 0x00a0},
      		{0x020c, 0x0190, 0x00a1},
      		{0x0000, 0x0000, 0x00c8},
      		{0x040c, 0x01a0, 0x00c9},
      		{0x0000, 0x0000, 0x00d0},
      		{0x0012, 0x0010, 0x00d1},
      		{0x0000, 0x0000, 0x00e8},
      		{0x0800, 0x0000, 0x0110},
      		{0x0001, 0x0000, 0x0119},
      		{0x0000, 0x0c00, 0x01a0},
      		{0x0000, 0x0100, 0x01a8},
      		{0x0000, 0x0000, 0x01f0},
      		{0x000c, 0x0230, 0x01f1},
      		{0x0200, 0x0000, 0x0238},
      		{0x0001, 0x0000, 0x0239},
      		{0x0000, 0x0000, 0x0240},
      		{0x0000, 0x0000, 0x0260},
      		{0x0012, 0x0040, 0x0261},
      		{0x0000, 0x0000, 0x0268},
      		{0x020c, 0x01b0, 0x0269},
      		{0x0000, 0x0000, 0x0280},
      		{0x0012, 0x0040, 0x0281},
      		{0x0002, 0x0000, 0x02b0},
      		{0x0000, 0x0000, 0x02b8},
      		{0x2000, 0x0000, 0x02c8},
      		{0x2000, 0x0000, 0x02f0},
      		{0x0041, 0x0100, 0x02f1},
      		{0x0000, 0x0100, 0x02f2},
      		{0x0000, 0x0000, 0x02f3},
      		{0x3000, 0x0000, 0x0300},
      		{0x0041, 0x0100, 0x0301},
      		{0x0000, 0x8b00, 0x0302},
      		{0x0000, 0x0000, 0x0303},
      		{0x2800, 0x0000, 0x0308},
      		{0x0041, 0x0100, 0x0309},
      		{0x0000, 0x8b00, 0x030a},
      		{0x0200, 0x0000, 0x030b},
      		{0x2000, 0x0000, 0x0310},
      		{0x0041, 0x0100, 0x0311},
      		{0x0000, 0x0100, 0x0312},
      		{0x0000, 0x0000, 0x0313},
      	};
      
      load_cfg((void*)cin, 0x20000, 378, 0, 0);
      config(0x0, 63, 0, 0);
      execute(0xe8a3, 0, 0);
      /// ADORA.BlockStore %1, %arg1 [0, %c0, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_1 = 0;
      uint64_t spadoffset_1 = 0;
      uint64_t roffset_1 = 0;
      store(arg_1 + dramoffset_1 + roffset_1, 0x1a000 + spadoffset_1, 8192, 0, 0);
      spadoffset_1 = spadoffset_1 + 8192;
      
      /// ADORA.BlockStore %3, %arg1 [0, %c32_0, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_3 = 256 * 32;
      uint64_t spadoffset_3 = 0;
      uint64_t roffset_3 = 0;
      store(arg_1 + dramoffset_3 + roffset_3, 0xa000 + spadoffset_3, 8192, 0, 0);
      spadoffset_3 = spadoffset_3 + 8192;
      
      /// ADORA.BlockStore %5, %arg1 [0, %c64_1, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "5", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_5 = 256 * 64;
      uint64_t spadoffset_5 = 0;
      uint64_t roffset_5 = 0;
      store(arg_1 + dramoffset_5 + roffset_5, 0x2000 + spadoffset_5, 8192, 0, 0);
      spadoffset_5 = spadoffset_5 + 8192;
      
      /// ADORA.BlockStore %7, %arg1 [0, %c96_2, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "7", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_7 = 256 * 96;
      uint64_t spadoffset_7 = 0;
      uint64_t roffset_7 = 0;
      store(arg_1 + dramoffset_7 + roffset_7, 0x1c000 + spadoffset_7, 8192, 0, 0);
      spadoffset_7 = spadoffset_7 + 8192;
      
      }
}
