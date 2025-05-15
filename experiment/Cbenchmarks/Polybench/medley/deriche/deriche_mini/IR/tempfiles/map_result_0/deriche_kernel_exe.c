
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void kernel_deriche(void* arg_0 ,void* arg_1 ,void* arg_2 ,void* arg_3){
  float float_4;
  float_4 = 0;
  float float_5;
  float_5 = 0;
  float float_6;
  float_6 = 0;
  float float_7;
  float_7 = 0;
  float float_8;
  float_8 = 0;
  float float_9;
  float_9 = 0;
  float float_10;
  float_10 = 0;
  float float_11;
  float_11 = 0;
  float float_12;
  float_12 = 0;
  float float_13;
  float_13 = 0;
  for (int int_14 = 0; int_14 < 64; int_14 = int_14 + 32){
    {
    /// %1 = ADORA.BlockLoad %arg0 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_0"}
    uint64_t dramoffset_0 = 256 * int_14;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 8192, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 8192;
    
    }
    {
    /// kernel_deriche_0
    volatile unsigned short cin[66][3] __attribute__((aligned(8))) = {
    		{0x2000, 0x0000, 0x0010},
    		{0x0041, 0x0100, 0x0011},
    		{0x0000, 0x0000, 0x0012},
    		{0x0000, 0x000a, 0x0013},
    		{0x0800, 0x0000, 0x0018},
    		{0x0001, 0x0100, 0x0019},
    		{0x0000, 0x0000, 0x001a},
    		{0x0000, 0x088a, 0x001b},
    		{0x0020, 0x0000, 0x001c},
    		{0x2800, 0x0000, 0x0028},
    		{0x0041, 0x0100, 0x0029},
    		{0x0000, 0x0000, 0x002a},
    		{0x0000, 0x090a, 0x002b},
    		{0x0020, 0x0000, 0x002c},
    		{0x0000, 0x0000, 0x0030},
    		{0x0001, 0x0100, 0x0031},
    		{0x0000, 0x0000, 0x0032},
    		{0x0000, 0x090a, 0x0033},
    		{0x0000, 0x0000, 0x0034},
    		{0x0000, 0x0010, 0x0058},
    		{0x0000, 0x0000, 0x0060},
    		{0x1000, 0x0000, 0x0068},
    		{0x1000, 0x0004, 0x0070},
    		{0x0011, 0x0020, 0x0078},
    		{0x44fd, 0x3f57, 0x00a8},
    		{0x0008, 0x0010, 0x00a9},
    		{0x0050, 0x0010, 0x00b1},
    		{0x0000, 0x5000, 0x00b2},
    		{0x0300, 0x8010, 0x00b3},
    		{0x0000, 0x0000, 0x00b4},
    		{0x200a, 0x0118, 0x00b9},
    		{0x0003, 0x0000, 0x00e9},
    		{0x2000, 0x0008, 0x00f0},
    		{0x0000, 0x0000, 0x00f9},
    		{0x00c0, 0x000c, 0x0100},
    		{0x0004, 0x0100, 0x0108},
    		{0x0150, 0x0010, 0x0131},
    		{0x0000, 0x5000, 0x0132},
    		{0x0000, 0x8010, 0x0133},
    		{0x0000, 0x0000, 0x0134},
    		{0x35c4, 0xbe41, 0x0138},
    		{0x0008, 0x0008, 0x0139},
    		{0x0050, 0x0010, 0x0141},
    		{0x0000, 0x5000, 0x0142},
    		{0x0000, 0x8010, 0x0143},
    		{0x0000, 0x0000, 0x0144},
    		{0x4598, 0xbf1b, 0x0148},
    		{0x0008, 0x0018, 0x0149},
    		{0x0003, 0x0000, 0x0179},
    		{0x0000, 0x2000, 0x0180},
    		{0x0000, 0x0000, 0x0181},
    		{0xd000, 0x0000, 0x0188},
    		{0x0000, 0x0400, 0x0190},
    		{0x010a, 0x00a0, 0x01c1},
    		{0x000a, 0x0048, 0x01d1},
    		{0x0003, 0x0000, 0x0209},
    		{0x2000, 0x0000, 0x0210},
    		{0xb54c, 0x3de1, 0x0258},
    		{0x0008, 0x0008, 0x0259},
    		{0x0000, 0x0030, 0x0298},
    		{0x0200, 0x0000, 0x02a0},
    		{0x0000, 0x0000, 0x02e8},
    		{0x0001, 0x0100, 0x02e9},
    		{0x0000, 0x0000, 0x02ea},
    		{0x0000, 0x08ca, 0x02eb},
    		{0x0000, 0x0000, 0x02ec},
    	};
    
    load_cfg((void*)cin, 0x20000, 396, 0, 0);
    config(0x0, 66, 0, 0);
    execute(0x436, 0, 0);
    }
    {
    /// ADORA.BlockStore %5, %alloca_12 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_0"}
    store(&float_11, 0x8000, 8, 0, 0);

    }
    {
    /// ADORA.BlockStore %4, %alloca_11 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_0"}
    store(&float_10, 0x2000, 8, 0, 0);

    }
    {
    /// ADORA.BlockStore %3, %alloca_14 [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_0"}
    store(&float_13, 0x10000, 8, 0, 0);

    }
    {
    /// ADORA.BlockStore %2, %arg2 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_0"}
    uint64_t dramoffset_1 = 256 * int_14;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    store(arg_2 + dramoffset_1 + roffset_1, 0xa000 + spadoffset_1, 8192, 0, 0);
    spadoffset_1 = spadoffset_1 + 8192;
    
    }
  }
  
  
  
}
