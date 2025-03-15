
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void forward_kernel_0(void* arg_0 ,void* arg_1 ,void* arg_2){
  for (int int_3 = 0; int_3 < 64; int_3 = int_3 + 1){
    for (int int_4 = 0; int_4 < 64; int_4 = int_4 + 32){
      /// %0 = ADORA.BlockLoad %arg2 [%arg3, %arg4] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_0 = 256 * int_3 + 4 * int_4;
      uint64_t spadoffset_0 = 0;
      for(int idx_0 = 0; idx_0 < 1; idx_0++){
        uint64_t roffset_0 =  256*idx_0 ;
        load_data(arg_2 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 32, 0, 0, 0);
        spadoffset_0 = spadoffset_0 + 32;
      } 
      /// %1 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_1 = 512 * int_3;
      uint64_t spadoffset_1 = 0;
      uint64_t roffset_1 = 0;
      load_data(arg_0 + dramoffset_1 + roffset_1, 0x18000 + spadoffset_1, 512, 0, 0, 0);
      spadoffset_1 = spadoffset_1 + 512;
      
      /// %2 = ADORA.BlockLoad %arg1 [0, %arg4] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "2", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_2 = 4 * int_4;
      uint64_t spadoffset_2 = 0;
      for(int idx_0 = 0; idx_0 < 128; idx_0++){
        uint64_t roffset_2 =  256*idx_0 ;
        load_data(arg_1 + dramoffset_2 + roffset_2, 0x1a000 + spadoffset_2, 32, 0, 0, 0);
        spadoffset_2 = spadoffset_2 + 32;
      } 
      int int_5 = int_4 + 8;
      /// %5 = ADORA.BlockLoad %arg2 [%arg3, %4] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "4", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_4 = 256 * int_3 + 4 * int_5;
      uint64_t spadoffset_4 = 0;
      for(int idx_0 = 0; idx_0 < 1; idx_0++){
        uint64_t roffset_4 =  256*idx_0 ;
        load_data(arg_2 + dramoffset_4 + roffset_4, 0x10000 + spadoffset_4, 32, 0, 0, 0);
        spadoffset_4 = spadoffset_4 + 32;
      } 
      /// %6 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "5", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_5 = 512 * int_3;
      uint64_t spadoffset_5 = 0;
      uint64_t roffset_5 = 0;
      load_data(arg_0 + dramoffset_5 + roffset_5, 0x2000 + spadoffset_5, 512, 0, 0, 0);
      spadoffset_5 = spadoffset_5 + 512;
      
      /// %7 = ADORA.BlockLoad %arg1 [0, %4] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "6", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_6 = 4 * int_5;
      uint64_t spadoffset_6 = 0;
      for(int idx_0 = 0; idx_0 < 128; idx_0++){
        uint64_t roffset_6 =  256*idx_0 ;
        load_data(arg_1 + dramoffset_6 + roffset_6, 0x8000 + spadoffset_6, 32, 0, 0, 0);
        spadoffset_6 = spadoffset_6 + 32;
      } 
      int int_6 = int_4 + 16;
      /// %10 = ADORA.BlockLoad %arg2 [%arg3, %9] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "8", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_8 = 256 * int_3 + 4 * int_6;
      uint64_t spadoffset_8 = 0;
      for(int idx_0 = 0; idx_0 < 1; idx_0++){
        uint64_t roffset_8 =  256*idx_0 ;
        load_data(arg_2 + dramoffset_8 + roffset_8, 0x12000 + spadoffset_8, 32, 0, 0, 0);
        spadoffset_8 = spadoffset_8 + 32;
      } 
      /// %11 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "9", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_9 = 512 * int_3;
      uint64_t spadoffset_9 = 0;
      uint64_t roffset_9 = 0;
      load_data(arg_0 + dramoffset_9 + roffset_9, 0x1c000 + spadoffset_9, 512, 0, 0, 0);
      spadoffset_9 = spadoffset_9 + 512;
      
      /// %12 = ADORA.BlockLoad %arg1 [0, %9] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "10", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_10 = 4 * int_6;
      uint64_t spadoffset_10 = 0;
      for(int idx_0 = 0; idx_0 < 128; idx_0++){
        uint64_t roffset_10 =  256*idx_0 ;
        load_data(arg_1 + dramoffset_10 + roffset_10, 0xa000 + spadoffset_10, 32, 0, 0, 0);
        spadoffset_10 = spadoffset_10 + 32;
      } 
      int int_7 = int_4 + 24;
      /// %15 = ADORA.BlockLoad %arg2 [%arg3, %14] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "12", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_12 = 256 * int_3 + 4 * int_7;
      uint64_t spadoffset_12 = 0;
      for(int idx_0 = 0; idx_0 < 1; idx_0++){
        uint64_t roffset_12 =  256*idx_0 ;
        load_data(arg_2 + dramoffset_12 + roffset_12, 0x4000 + spadoffset_12, 32, 0, 0, 0);
        spadoffset_12 = spadoffset_12 + 32;
      } 
      /// %16 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "13", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_13 = 512 * int_3;
      uint64_t spadoffset_13 = 0;
      uint64_t roffset_13 = 0;
      load_data(arg_0 + dramoffset_13 + roffset_13, 0x6000 + spadoffset_13, 512, 0, 0, 0);
      spadoffset_13 = spadoffset_13 + 512;
      
      /// %17 = ADORA.BlockLoad %arg1 [0, %14] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "14", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_14 = 4 * int_7;
      uint64_t spadoffset_14 = 0;
      for(int idx_0 = 0; idx_0 < 128; idx_0++){
        uint64_t roffset_14 =  256*idx_0 ;
        load_data(arg_1 + dramoffset_14 + roffset_14, 0xc000 + spadoffset_14, 32, 0, 0, 0);
        spadoffset_14 = spadoffset_14 + 32;
      } 
      volatile unsigned short cin[120][3] __attribute__((aligned(8))) = {
      		{0x2800, 0x0000, 0x0008},
      		{0xe042, 0x0047, 0x0009},
      		{0x0000, 0x0100, 0x000a},
      		{0x0000, 0x0000, 0x000b},
      		{0x0000, 0x0000, 0x0010},
      		{0x0042, 0x0040, 0x0011},
      		{0x0000, 0x0300, 0x0012},
      		{0x0000, 0x0000, 0x0013},
      		{0x3800, 0x0000, 0x0018},
      		{0xe042, 0x0047, 0x0019},
      		{0x0000, 0x0100, 0x001a},
      		{0x0000, 0x0000, 0x001b},
      		{0x1000, 0x0000, 0x0020},
      		{0x0042, 0x0040, 0x0021},
      		{0x0000, 0x0100, 0x0022},
      		{0x0000, 0x0000, 0x0023},
      		{0x1800, 0x0000, 0x0028},
      		{0x0042, 0x0040, 0x0029},
      		{0x0000, 0x9500, 0x002a},
      		{0x0000, 0x0000, 0x002b},
      		{0x0800, 0x0001, 0x0030},
      		{0x0242, 0x0047, 0x0031},
      		{0x0000, 0x0100, 0x0032},
      		{0x0000, 0x0000, 0x0033},
      		{0x1000, 0x0001, 0x0038},
      		{0x0242, 0x0047, 0x0039},
      		{0x0000, 0x0100, 0x003a},
      		{0x0000, 0x0000, 0x003b},
      		{0x0000, 0x0001, 0x0040},
      		{0x0242, 0x0047, 0x0041},
      		{0x0000, 0x0100, 0x0042},
      		{0x0000, 0x0000, 0x0043},
      		{0x0000, 0x0000, 0x0058},
      		{0x0000, 0x0000, 0x0060},
      		{0x0400, 0x0000, 0x0068},
      		{0x0110, 0x0000, 0x0070},
      		{0x8000, 0x0010, 0x0078},
      		{0x8000, 0x0000, 0x0080},
      		{0x0000, 0x0000, 0x0088},
      		{0x080e, 0x0028, 0x00a9},
      		{0x0a0e, 0x0018, 0x00b1},
      		{0x020d, 0x0022, 0x00b9},
      		{0x040d, 0x0026, 0x00c1},
      		{0x0000, 0x6000, 0x00e8},
      		{0x0004, 0x0000, 0x00e9},
      		{0x0000, 0x4000, 0x00f0},
      		{0x0000, 0x4000, 0x00f8},
      		{0x0001, 0x4000, 0x0100},
      		{0x0080, 0x0180, 0x0108},
      		{0x0003, 0x0000, 0x0109},
      		{0x0000, 0x0200, 0x0110},
      		{0x0010, 0x0004, 0x0141},
      		{0x0000, 0x0400, 0x0142},
      		{0x0180, 0x0108, 0x0143},
      		{0x0000, 0x0000, 0x0144},
      		{0x0010, 0x0004, 0x0149},
      		{0x0000, 0x0400, 0x014a},
      		{0x0200, 0x0108, 0x014b},
      		{0x0000, 0x0000, 0x014c},
      		{0x0003, 0x0000, 0x0179},
      		{0x0001, 0x0000, 0x0191},
      		{0x0000, 0x8000, 0x0198},
      		{0x0003, 0x0000, 0x0199},
      		{0x1000, 0x0000, 0x01a0},
      		{0x080e, 0x0038, 0x01d1},
      		{0x0010, 0x0002, 0x01e9},
      		{0x0000, 0x0400, 0x01ea},
      		{0x01c0, 0x0108, 0x01eb},
      		{0x0000, 0x0000, 0x01ec},
      		{0x0003, 0x0000, 0x0209},
      		{0x0200, 0x0000, 0x0218},
      		{0x0000, 0x0008, 0x0220},
      		{0x0000, 0x0000, 0x0221},
      		{0xc000, 0x0c04, 0x0228},
      		{0x0000, 0x0080, 0x0230},
      		{0x0a0e, 0x0034, 0x0261},
      		{0x0010, 0x0004, 0x0269},
      		{0x0000, 0x0400, 0x026a},
      		{0x01c0, 0x0108, 0x026b},
      		{0x0000, 0x0000, 0x026c},
      		{0x004d, 0x0018, 0x0271},
      		{0x000d, 0x0038, 0x0279},
      		{0x2000, 0x0000, 0x0298},
      		{0x0000, 0x0010, 0x02a0},
      		{0x0040, 0x0000, 0x02a8},
      		{0x0300, 0x0000, 0x02b0},
      		{0x0000, 0x0000, 0x02c0},
      		{0x0000, 0x0000, 0x02c8},
      		{0x1800, 0x0000, 0x02d8},
      		{0x0042, 0x0040, 0x02d9},
      		{0x0000, 0x9500, 0x02da},
      		{0x0200, 0x0000, 0x02db},
      		{0x0000, 0x0000, 0x02e0},
      		{0x0042, 0x0040, 0x02e1},
      		{0x0000, 0x0300, 0x02e2},
      		{0x0000, 0x0000, 0x02e3},
      		{0x1000, 0x0000, 0x02e8},
      		{0x0042, 0x0040, 0x02e9},
      		{0x0000, 0x9500, 0x02ea},
      		{0x0200, 0x0000, 0x02eb},
      		{0x0800, 0x0000, 0x02f0},
      		{0x0042, 0x0040, 0x02f1},
      		{0x0000, 0x0100, 0x02f2},
      		{0x0000, 0x0000, 0x02f3},
      		{0x1800, 0x0000, 0x02f8},
      		{0x0042, 0x0040, 0x02f9},
      		{0x0000, 0x9500, 0x02fa},
      		{0x0000, 0x0000, 0x02fb},
      		{0x0800, 0x0001, 0x0300},
      		{0x0242, 0x0047, 0x0301},
      		{0x0000, 0x0100, 0x0302},
      		{0x0000, 0x0000, 0x0303},
      		{0x3000, 0x0000, 0x0308},
      		{0xe042, 0x0047, 0x0309},
      		{0x0000, 0x0100, 0x030a},
      		{0x0000, 0x0000, 0x030b},
      		{0x2000, 0x0000, 0x0310},
      		{0xe042, 0x0047, 0x0311},
      		{0x0000, 0x0100, 0x0312},
      		{0x0000, 0x0000, 0x0313},
      	};
      
      load_cfg((void*)cin, 0x20000, 720, 0, 0);
      config(0x0, 120, 0, 0);
      execute(0xffff, 0, 0);
      /// ADORA.BlockStore %3, %arg2 [%arg3, %arg4] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_3 = 32 * int_3 + 4 * int_4;
      uint64_t spadoffset_3 = 0;
      uint64_t roffset_3 = 0;
      store(arg_2 + dramoffset_3 + roffset_3, 0xe000 + spadoffset_3, 32, 0, 0);
      spadoffset_3 = spadoffset_3 + 32;
      
      int int_8 = int_4 + 8;
      /// ADORA.BlockStore %8, %arg2 [%arg3, %19] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "7", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_7 = 32 * int_3 + 4 * int_8;
      uint64_t spadoffset_7 = 0;
      uint64_t roffset_7 = 0;
      store(arg_2 + dramoffset_7 + roffset_7, 0x14000 + spadoffset_7, 32, 0, 0);
      spadoffset_7 = spadoffset_7 + 32;
      
      int int_9 = int_4 + 16;
      /// ADORA.BlockStore %13, %arg2 [%arg3, %20] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "11", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_11 = 32 * int_3 + 4 * int_9;
      uint64_t spadoffset_11 = 0;
      uint64_t roffset_11 = 0;
      store(arg_2 + dramoffset_11 + roffset_11, 0x1e000 + spadoffset_11, 32, 0, 0);
      spadoffset_11 = spadoffset_11 + 32;
      
      int int_10 = int_4 + 24;
      /// ADORA.BlockStore %18, %arg2 [%arg3, %21] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "15", KernelName = "forward_kernel_0"}
      uint64_t dramoffset_15 = 32 * int_3 + 4 * int_10;
      uint64_t spadoffset_15 = 0;
      uint64_t roffset_15 = 0;
      store(arg_2 + dramoffset_15 + roffset_15, 0x16000 + spadoffset_15, 32, 0, 0);
      spadoffset_15 = spadoffset_15 + 32;
      
    }
  }
}
