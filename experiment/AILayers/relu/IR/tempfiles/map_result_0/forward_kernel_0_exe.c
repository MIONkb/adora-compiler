
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void forward_kernel_0(void* arg_0 ,void* arg_1){
    for (int int_2 = 0; int_2 < 1; int_2 = int_2 + 1){
      {
      /// %0 = ADORA.BlockLoad %arg0 [0, %c0, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_0 = 0;
      uint64_t spadoffset_0 = 0;
      uint64_t roffset_0 = 0;
      load_data(arg_0 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 8192, 0, 0, 0);
      spadoffset_0 = spadoffset_0 + 8192;
      
      }
      {
      /// %2 = ADORA.BlockLoad %arg0 [0, %c32, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "2", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_2 = 256 * 32;
      uint64_t spadoffset_2 = 0;
      uint64_t roffset_2 = 0;
      load_data(arg_0 + dramoffset_2 + roffset_2, 0x0 + spadoffset_2, 8192, 0, 0, 0);
      spadoffset_2 = spadoffset_2 + 8192;
      
      }
      {
      /// %4 = ADORA.BlockLoad %arg0 [0, %c64, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "4", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_4 = 256 * 64;
      uint64_t spadoffset_4 = 0;
      uint64_t roffset_4 = 0;
      load_data(arg_0 + dramoffset_4 + roffset_4, 0x8000 + spadoffset_4, 8192, 0, 0, 0);
      spadoffset_4 = spadoffset_4 + 8192;
      
      }
      {
      /// %6 = ADORA.BlockLoad %arg0 [0, %c96, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "6", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_6 = 256 * 96;
      uint64_t spadoffset_6 = 0;
      uint64_t roffset_6 = 0;
      load_data(arg_0 + dramoffset_6 + roffset_6, 0x10000 + spadoffset_6, 8192, 0, 0, 0);
      spadoffset_6 = spadoffset_6 + 8192;
      
      }
      {
      /// forward_kernel_0
      volatile unsigned short cin[57][3] __attribute__((aligned(8))) = {
      		{0x2000, 0x0000, 0x0008},
      		{0x0041, 0x0100, 0x0009},
      		{0x0000, 0x0100, 0x000a},
      		{0x0000, 0x0000, 0x000b},
      		{0x2800, 0x0000, 0x0018},
      		{0x0041, 0x0100, 0x0019},
      		{0x0000, 0x5100, 0x001a},
      		{0x0004, 0x0000, 0x001b},
      		{0x2800, 0x0000, 0x0038},
      		{0x0041, 0x0100, 0x0039},
      		{0x0000, 0x5100, 0x003a},
      		{0x1004, 0x0000, 0x003b},
      		{0x2000, 0x0000, 0x0040},
      		{0x0041, 0x0100, 0x0041},
      		{0x0000, 0x0100, 0x0042},
      		{0x0000, 0x0000, 0x0043},
      		{0x0000, 0x0000, 0x0058},
      		{0x0010, 0x0000, 0x0060},
      		{0x0200, 0x0000, 0x0088},
      		{0x0000, 0x0000, 0x0098},
      		{0x000d, 0x0040, 0x0099},
      		{0x0000, 0x0000, 0x00a0},
      		{0x0807, 0x0640, 0x00a1},
      		{0x0000, 0x0000, 0x00c8},
      		{0x000d, 0x0080, 0x00c9},
      		{0x0000, 0x0000, 0x00d0},
      		{0x0807, 0x0640, 0x00d1},
      		{0x0000, 0x0000, 0x00e8},
      		{0x0000, 0x0000, 0x0118},
      		{0x0000, 0x0000, 0x0250},
      		{0x000d, 0x0100, 0x0251},
      		{0x0000, 0x0000, 0x0258},
      		{0x0807, 0x0700, 0x0259},
      		{0x0000, 0x0000, 0x0270},
      		{0x000d, 0x0100, 0x0271},
      		{0x0000, 0x0000, 0x0278},
      		{0x0807, 0x0700, 0x0279},
      		{0x0000, 0x0000, 0x02a0},
      		{0x0002, 0x0000, 0x02a8},
      		{0x0000, 0x0000, 0x02c0},
      		{0x0002, 0x0000, 0x02c8},
      		{0x2800, 0x0000, 0x02e0},
      		{0x0041, 0x0100, 0x02e1},
      		{0x0000, 0x5100, 0x02e2},
      		{0x1004, 0x0000, 0x02e3},
      		{0x2000, 0x0000, 0x02e8},
      		{0x0041, 0x0100, 0x02e9},
      		{0x0000, 0x0100, 0x02ea},
      		{0x0000, 0x0000, 0x02eb},
      		{0x2800, 0x0000, 0x0300},
      		{0x0041, 0x0100, 0x0301},
      		{0x0000, 0x5100, 0x0302},
      		{0x1004, 0x0000, 0x0303},
      		{0x2000, 0x0000, 0x0308},
      		{0x0041, 0x0100, 0x0309},
      		{0x0000, 0x0100, 0x030a},
      		{0x0000, 0x0000, 0x030b},
      	};
      
      load_cfg((void*)cin, 0x20000, 342, 0, 0);
      config(0x0, 57, 0, 0);
      execute(0x66c5, 0, 0);
      }
      {
      /// ADORA.BlockStore %1, %arg1 [0, %c0, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_1 = 0;
      uint64_t spadoffset_1 = 0;
      uint64_t roffset_1 = 0;
      store(arg_1 + dramoffset_1 + roffset_1, 0x1a000 + spadoffset_1, 8192, 0, 0);
      spadoffset_1 = spadoffset_1 + 8192;
      
      }
      {
      /// ADORA.BlockStore %3, %arg1 [0, %c32_0, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_3 = 256 * 32;
      uint64_t spadoffset_3 = 0;
      uint64_t roffset_3 = 0;
      store(arg_1 + dramoffset_3 + roffset_3, 0x2000 + spadoffset_3, 8192, 0, 0);
      spadoffset_3 = spadoffset_3 + 8192;
      
      }
      {
      /// ADORA.BlockStore %5, %arg1 [0, %c64_1, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "5", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_5 = 256 * 64;
      uint64_t spadoffset_5 = 0;
      uint64_t roffset_5 = 0;
      store(arg_1 + dramoffset_5 + roffset_5, 0xa000 + spadoffset_5, 8192, 0, 0);
      spadoffset_5 = spadoffset_5 + 8192;
      
      }
      {
      /// ADORA.BlockStore %7, %arg1 [0, %c96_2, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "7", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_7 = 256 * 96;
      uint64_t spadoffset_7 = 0;
      uint64_t roffset_7 = 0;
      store(arg_1 + dramoffset_7 + roffset_7, 0x12000 + spadoffset_7, 8192, 0, 0);
      spadoffset_7 = spadoffset_7 + 8192;
      
      }
    }
    
    
    
}
