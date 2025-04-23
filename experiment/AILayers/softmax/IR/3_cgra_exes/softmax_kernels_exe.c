
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void forward(void* arg_0 ,void* arg_1 ,void* arg_2 ,void* arg_3 ,void* arg_4 ,void* arg_5 ,void* arg_6 ,void* arg_7){
  for (int int_8 = 0; int_8 < 1; int_8 = int_8 + 1){
    for (int int_9 = 0; int_9 < 128; int_9 = int_9 + 32){
      {
      /// %0 = ADORA.BlockLoad %arg1 [%arg8, %arg9, 0] : memref<1x128x1xf32> -> memref<1x32x1xf32>  {Id = "0", KernelName = "forward_0"}
      uint64_t dramoffset_0 = 512 * int_8 + 4 * int_9;
      uint64_t spadoffset_0 = 0;
      uint64_t roffset_0 = 0;
      spadoffset_0 = spadoffset_0 + 128;
      
      }
      {
      /// %1 = ADORA.BlockLoad %arg0 [%arg8, %arg9, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "1", KernelName = "forward_0"}
      uint64_t dramoffset_1 = 32768 * int_8 + 256 * int_9;
      uint64_t spadoffset_1 = 0;
      uint64_t roffset_1 = 0;
      load_data(arg_0 + dramoffset_1 + roffset_1, 0x18000 + spadoffset_1, 8192, 0, 0, 0);
      spadoffset_1 = spadoffset_1 + 8192;
      
      }
      {
      /// forward_0
      volatile unsigned short cin[26][3] __attribute__((aligned(8))) = {
      		{0x0004, 0x0000, 0x00f1},
      		{0x0000, 0x0200, 0x00f8},
      		{0x0000, 0x0002, 0x0180},
      		{0x0000, 0x000c, 0x0188},
      		{0x0000, 0x0080, 0x0190},
      		{0xa007, 0x0310, 0x01c9},
      		{0x0000, 0x3fc0, 0x01d0},
      		{0x000d, 0x0100, 0x01d1},
      		{0x0987, 0x08d8, 0x01d9},
      		{0x0000, 0x0000, 0x0218},
      		{0x0000, 0x0000, 0x0219},
      		{0x0244, 0x0000, 0x0220},
      		{0x0000, 0x0000, 0x0228},
      		{0x0050, 0x0008, 0x0261},
      		{0x010d, 0x0118, 0x0269},
      		{0x0300, 0x0000, 0x02a8},
      		{0x0000, 0x0000, 0x02b0},
      		{0x0002, 0x0000, 0x02b8},
      		{0x0000, 0x0000, 0x02f0},
      		{0x0041, 0x0100, 0x02f1},
      		{0x0000, 0x6100, 0x02f2},
      		{0x0004, 0x0000, 0x02f3},
      		{0x2800, 0x0000, 0x02f8},
      		{0x0041, 0x0100, 0x02f9},
      		{0x0000, 0x0100, 0x02fa},
      		{0x0000, 0x0000, 0x02fb},
      	};
      
      load_cfg((void*)cin, 0x20000, 156, 0, 0);
      config(0x0, 26, 0, 0);
      execute(0xfc4d, 0, 0);
      }
      {
      /// ADORA.BlockStore %2, %arg1 [0, %arg9, 0] : memref<1x32x1xf32> -> memref<1x128x1xf32>  {Id = "2", KernelName = "forward_0"}
      uint64_t dramoffset_2 = 4 * int_9;
      uint64_t spadoffset_2 = 0;
      uint64_t roffset_2 = 0;
      store(arg_1 + dramoffset_2 + roffset_2, 0x10000 + spadoffset_2, 128, 0, 0);
      spadoffset_2 = spadoffset_2 + 128;
      
      }
    }
    
    
    
  }
  
  
  
  for (int int_10 = 0; int_10 < 1; int_10 = int_10 + 1){
    for (int int_11 = 0; int_11 < 128; int_11 = int_11 + 32){
      {
      /// %0 = ADORA.BlockLoad %arg0 [0, %arg9, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "0", KernelName = "forward_1"}
      uint64_t dramoffset_0 = 256 * int_11;
      uint64_t spadoffset_0 = 0;
      uint64_t roffset_0 = 0;
      load_data(arg_0 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 8192, 0, 0, 0);
      spadoffset_0 = spadoffset_0 + 8192;
      
      }
      {
      /// %1 = ADORA.BlockLoad %arg1 [0, %arg9, 0] : memref<1x128x1xf32> -> memref<1x32x1xf32>  {Id = "1", KernelName = "forward_1"}
      uint64_t dramoffset_1 = 4 * int_11;
      uint64_t spadoffset_1 = 0;
      uint64_t roffset_1 = 0;
      load_data(arg_1 + dramoffset_1 + roffset_1, 0x10000 + spadoffset_1, 128, 0, 0, 0);
      spadoffset_1 = spadoffset_1 + 128;
      
      }
      {
      /// forward_1
      volatile unsigned short cin[15][3] __attribute__((aligned(8))) = {
      		{0x000a, 0x00e0, 0x0269},
      		{0x0100, 0x0000, 0x02b0},
      		{0x0000, 0x0000, 0x02b8},
      		{0x0800, 0x0000, 0x02f0},
      		{0x0041, 0x0100, 0x02f1},
      		{0x0000, 0x0100, 0x02f2},
      		{0x0000, 0x0000, 0x02f3},
      		{0x3000, 0x0000, 0x02f8},
      		{0x0041, 0x0100, 0x02f9},
      		{0x0000, 0x4100, 0x02fa},
      		{0x0004, 0x0000, 0x02fb},
      		{0x2000, 0x0000, 0x0300},
      		{0x0041, 0x0100, 0x0301},
      		{0x0000, 0x0100, 0x0302},
      		{0x0000, 0x0000, 0x0303},
      	};
      
      load_cfg((void*)cin, 0x20000, 90, 0, 0);
      config(0x0, 15, 0, 0);
      execute(0xfc4d, 0, 0);
      }
      {
      /// ADORA.BlockStore %2, %arg3 [0, %arg9, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "2", KernelName = "forward_1"}
      uint64_t dramoffset_2 = 256 * int_11;
      uint64_t spadoffset_2 = 0;
      uint64_t roffset_2 = 0;
      store(arg_3 + dramoffset_2 + roffset_2, 0x1a000 + spadoffset_2, 8192, 0, 0);
      spadoffset_2 = spadoffset_2 + 8192;
      
      }
    }
    
    
    
  }
  
  
  
  for (int int_12 = 0; int_12 < 1; int_12 = int_12 + 1){
    for (int int_13 = 0; int_13 < 128; int_13 = int_13 + 32){
      {
      /// %0 = ADORA.BlockLoad %arg3 [0, %arg9, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "0", KernelName = "forward_2"}
      uint64_t dramoffset_0 = 256 * int_13;
      uint64_t spadoffset_0 = 0;
      uint64_t roffset_0 = 0;
      load_data(arg_3 + dramoffset_0 + roffset_0, 0x10000 + spadoffset_0, 8192, 0, 0, 0);
      spadoffset_0 = spadoffset_0 + 8192;
      
      }
      {
      /// forward_2
      volatile unsigned short cin[28][3] __attribute__((aligned(8))) = {
      		{0x2000, 0x0000, 0x0020},
      		{0x0041, 0x0100, 0x0021},
      		{0x0000, 0xc100, 0x0022},
      		{0x0004, 0x0000, 0x0023},
      		{0x0030, 0x0000, 0x0068},
      		{0x0000, 0x0400, 0x00f8},
      		{0x1808, 0x00e0, 0x0139},
      		{0x0000, 0x0000, 0x0180},
      		{0x0002, 0x0000, 0x0188},
      		{0x0008, 0x0018, 0x01c1},
      		{0x1008, 0x0118, 0x01c9},
      		{0x0040, 0x0010, 0x0208},
      		{0x0000, 0x0000, 0x0210},
      		{0x0004, 0x0000, 0x0218},
      		{0x59df, 0x5f37, 0x0248},
      		{0x0002, 0x0080, 0x0249},
      		{0x0001, 0x0000, 0x0250},
      		{0x0005, 0x0020, 0x0251},
      		{0x0000, 0x3f00, 0x0258},
      		{0x0008, 0x0100, 0x0259},
      		{0x0000, 0x3fc0, 0x0260},
      		{0x000a, 0x0040, 0x0261},
      		{0x0000, 0x0000, 0x02a0},
      		{0x0002, 0x0000, 0x02a8},
      		{0x2000, 0x0000, 0x02e8},
      		{0x0041, 0x0100, 0x02e9},
      		{0x0000, 0x0100, 0x02ea},
      		{0x0000, 0x0000, 0x02eb},
      	};
      
      load_cfg((void*)cin, 0x20000, 168, 0, 0);
      config(0x0, 28, 0, 0);
      execute(0xfc4d, 0, 0);
      }
      {
      /// ADORA.BlockStore %1, %arg4 [0, %arg9, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "1", KernelName = "forward_2"}
      uint64_t dramoffset_1 = 256 * int_13;
      uint64_t spadoffset_1 = 0;
      uint64_t roffset_1 = 0;
      store(arg_4 + dramoffset_1 + roffset_1, 0x0 + spadoffset_1, 8192, 0, 0);
      spadoffset_1 = spadoffset_1 + 8192;
      
      }
    }
    
    
    
  }
  
  
  
  for (int int_14 = 0; int_14 < 1; int_14 = int_14 + 1){
    for (int int_15 = 0; int_15 < 128; int_15 = int_15 + 32){
      {
      /// %0 = ADORA.BlockLoad %arg6 [%arg8, %arg9, 0] : memref<1x128x1xf32> -> memref<1x32x1xf32>  {Id = "0", KernelName = "forward_3"}
      uint64_t dramoffset_0 = 512 * int_14 + 4 * int_15;
      uint64_t spadoffset_0 = 0;
      uint64_t roffset_0 = 0;
      load_data(arg_6 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 128, 0, 0, 0);
      spadoffset_0 = spadoffset_0 + 128;
      
      }
      {
      /// %1 = ADORA.BlockLoad %arg4 [%arg8, %arg9, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "1", KernelName = "forward_3"}
      uint64_t dramoffset_1 = 32768 * int_14 + 256 * int_15;
      uint64_t spadoffset_1 = 0;
      uint64_t roffset_1 = 0;
      load_data(arg_4 + dramoffset_1 + roffset_1, 0x8000 + spadoffset_1, 8192, 0, 0, 0);
      spadoffset_1 = spadoffset_1 + 8192;
      
      }
      {
      /// forward_3
      volatile unsigned short cin[23][3] __attribute__((aligned(8))) = {
      		{0x0000, 0x0000, 0x0008},
      		{0x0041, 0x0100, 0x0009},
      		{0x0000, 0x0100, 0x000a},
      		{0x0000, 0x0000, 0x000b},
      		{0x0800, 0x0000, 0x0018},
      		{0x0041, 0x0100, 0x0019},
      		{0x0000, 0x6100, 0x001a},
      		{0x1004, 0x0000, 0x001b},
      		{0x2000, 0x0000, 0x0038},
      		{0x0041, 0x0100, 0x0039},
      		{0x0000, 0x0100, 0x003a},
      		{0x0000, 0x0000, 0x003b},
      		{0x0000, 0x0000, 0x0058},
      		{0x0000, 0x0008, 0x0060},
      		{0x0400, 0x0000, 0x0068},
      		{0x2000, 0x0000, 0x0070},
      		{0x0000, 0x0001, 0x0078},
      		{0x0000, 0x0000, 0x0080},
      		{0x080b, 0x0050, 0x00b1},
      		{0x001b, 0x0010, 0x00c1},
      		{0x0000, 0x1000, 0x00c2},
      		{0x0300, 0x0410, 0x00c3},
      		{0x0000, 0x0000, 0x00c4},
      	};
      
      load_cfg((void*)cin, 0x20000, 138, 0, 0);
      config(0x0, 23, 0, 0);
      execute(0xfc4d, 0, 0);
      }
      {
      /// ADORA.BlockStore %2, %arg6 [0, %arg9, 0] : memref<1x32x1xf32> -> memref<1x128x1xf32>  {Id = "2", KernelName = "forward_3"}
      uint64_t dramoffset_2 = 4 * int_15;
      uint64_t spadoffset_2 = 0;
      uint64_t roffset_2 = 0;
      store(arg_6 + dramoffset_2 + roffset_2, 0x2000 + spadoffset_2, 128, 0, 0);
      spadoffset_2 = spadoffset_2 + 128;
      
      }
    }
    
    
    
  }
  
  
  
  for (int int_16 = 0; int_16 < 1; int_16 = int_16 + 1){
    for (int int_17 = 0; int_17 < 128; int_17 = int_17 + 32){
      {
      /// %0 = ADORA.BlockLoad %arg4 [0, %arg9, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "0", KernelName = "forward_4"}
      uint64_t dramoffset_0 = 256 * int_17;
      uint64_t spadoffset_0 = 0;
      uint64_t roffset_0 = 0;
      load_data(arg_4 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 8192, 0, 0, 0);
      spadoffset_0 = spadoffset_0 + 8192;
      
      }
      {
      /// %1 = ADORA.BlockLoad %arg6 [0, %arg9, 0] : memref<1x128x1xf32> -> memref<1x32x1xf32>  {Id = "1", KernelName = "forward_4"}
      uint64_t dramoffset_1 = 4 * int_17;
      uint64_t spadoffset_1 = 0;
      uint64_t roffset_1 = 0;
      load_data(arg_6 + dramoffset_1 + roffset_1, 0x1a000 + spadoffset_1, 128, 0, 0, 0);
      spadoffset_1 = spadoffset_1 + 128;
      
      }
      {
      /// forward_4
      volatile unsigned short cin[15][3] __attribute__((aligned(8))) = {
      		{0x0009, 0x0118, 0x0279},
      		{0x0110, 0x0000, 0x02c0},
      		{0x0000, 0x0000, 0x02c8},
      		{0x2000, 0x0000, 0x0300},
      		{0x0041, 0x0100, 0x0301},
      		{0x0000, 0x0100, 0x0302},
      		{0x0000, 0x0000, 0x0303},
      		{0x3000, 0x0000, 0x0308},
      		{0x0041, 0x0100, 0x0309},
      		{0x0000, 0x5100, 0x030a},
      		{0x0004, 0x0000, 0x030b},
      		{0x0800, 0x0000, 0x0310},
      		{0x0041, 0x0100, 0x0311},
      		{0x0000, 0x0100, 0x0312},
      		{0x0000, 0x0000, 0x0313},
      	};
      
      load_cfg((void*)cin, 0x20000, 90, 0, 0);
      config(0x0, 15, 0, 0);
      execute(0xfc4d, 0, 0);
      }
      {
      /// ADORA.BlockStore %2, %arg7 [0, %arg9, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "2", KernelName = "forward_4"}
      uint64_t dramoffset_2 = 256 * int_17;
      uint64_t spadoffset_2 = 0;
      uint64_t roffset_2 = 0;
      store(arg_7 + dramoffset_2 + roffset_2, 0x1c000 + spadoffset_2, 8192, 0, 0);
      spadoffset_2 = spadoffset_2 + 8192;
      
      }
    }
    
    
    
  }
  
  
  
}
