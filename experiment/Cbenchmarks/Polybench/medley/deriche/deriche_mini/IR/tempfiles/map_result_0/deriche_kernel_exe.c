
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
    volatile unsigned short cin[50][3] __attribute__((aligned(8))) = {
    		{0x0000, 0x0000, 0x0008},
    		{0x0001, 0x0100, 0x0009},
    		{0x0000, 0x5100, 0x000a},
    		{0x1004, 0x0000, 0x000b},
    		{0x2800, 0x0000, 0x0010},
    		{0x0041, 0x0100, 0x0011},
    		{0x0000, 0x0100, 0x0012},
    		{0x0000, 0x0000, 0x0013},
    		{0x1000, 0x0000, 0x0018},
    		{0x0001, 0x0100, 0x0019},
    		{0x0000, 0x5100, 0x001a},
    		{0x0004, 0x0000, 0x001b},
    		{0x2800, 0x0000, 0x0020},
    		{0x0041, 0x0100, 0x0021},
    		{0x0000, 0x8100, 0x0022},
    		{0x1004, 0x0000, 0x0023},
    		{0x0000, 0x0000, 0x0028},
    		{0x0001, 0x0100, 0x0029},
    		{0x0000, 0x8100, 0x002a},
    		{0x0004, 0x0000, 0x002b},
    		{0x0001, 0x0000, 0x0050},
    		{0x0003, 0x0000, 0x0058},
    		{0x0030, 0x0000, 0x0060},
    		{0x0011, 0x0000, 0x0070},
    		{0x35c4, 0xbe41, 0x0098},
    		{0x0008, 0x0010, 0x0099},
    		{0x0050, 0x0018, 0x00a1},
    		{0x44fd, 0x3f57, 0x00a8},
    		{0x0008, 0x0100, 0x00a9},
    		{0x018b, 0x0118, 0x00b1},
    		{0x0000, 0x0000, 0x00e0},
    		{0x4080, 0x0800, 0x00e8},
    		{0x0000, 0x0000, 0x00e9},
    		{0x0000, 0x0000, 0x00f0},
    		{0x00c4, 0x2000, 0x00f8},
    		{0x0004, 0x0000, 0x0100},
    		{0x0002, 0x0000, 0x0101},
    		{0xb54c, 0x3de1, 0x0130},
    		{0x0008, 0x0040, 0x0131},
    		{0x0050, 0x0010, 0x0139},
    		{0x000b, 0x0058, 0x0141},
    		{0x6000, 0x0000, 0x0178},
    		{0x0000, 0x2000, 0x0180},
    		{0x0080, 0x000c, 0x0188},
    		{0x0000, 0x0100, 0x0190},
    		{0x100b, 0x0048, 0x01c1},
    		{0x0050, 0x0010, 0x01c9},
    		{0x4598, 0xbf1b, 0x01d0},
    		{0x0008, 0x0018, 0x01d1},
    		{0x0000, 0x0000, 0x0218},
    	};
    
    load_cfg((void*)cin, 0x20000, 300, 0, 0);
    config(0x0, 50, 0, 0);
    execute(0x9fbf, 0, 0);
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
    store(&float_13, 0x4000, 8, 0, 0);

    }
    {
    /// ADORA.BlockStore %2, %arg2 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_0"}
    uint64_t dramoffset_1 = 256 * int_14;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    store(arg_2 + dramoffset_1 + roffset_1, 0x6000 + spadoffset_1, 8192, 0, 0);
    spadoffset_1 = spadoffset_1 + 8192;
    
    }
  }
  
  
  
  for (int int_15 = 0; int_15 < 64; int_15 = int_15 + 32){
    {
    /// %1 = ADORA.BlockLoad %arg0 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_1"}
    uint64_t dramoffset_0 = 256 * int_15;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 8192, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 8192;
    
    }
    {
    /// kernel_deriche_1
    volatile unsigned short cin[59][3] __attribute__((aligned(8))) = {
    		{0x0000, 0x0000, 0x0018},
    		{0x0001, 0x0100, 0x0019},
    		{0x0000, 0x7100, 0x001a},
    		{0x1004, 0x0000, 0x001b},
    		{0xe800, 0x03ff, 0x0020},
    		{0x1fc1, 0x0100, 0x0021},
    		{0x0000, 0x7100, 0x0022},
    		{0x1004, 0x0000, 0x0023},
    		{0x0000, 0x0000, 0x0028},
    		{0x0001, 0x0100, 0x0029},
    		{0x0000, 0x3100, 0x002a},
    		{0x1004, 0x0000, 0x002b},
    		{0x0800, 0x0000, 0x0040},
    		{0x0001, 0x0100, 0x0041},
    		{0x0000, 0x1100, 0x0042},
    		{0x0004, 0x0000, 0x0043},
    		{0x0000, 0x0000, 0x0068},
    		{0x0001, 0x0000, 0x0070},
    		{0x0001, 0x0000, 0x0078},
    		{0x0030, 0x0000, 0x0088},
    		{0x180b, 0x00e0, 0x00b1},
    		{0x0050, 0x0018, 0x00b9},
    		{0x0180, 0x0020, 0x00c1},
    		{0x0000, 0x0004, 0x00f8},
    		{0x0002, 0x0000, 0x0100},
    		{0x0000, 0x0000, 0x0108},
    		{0x0000, 0x0000, 0x0110},
    		{0x0000, 0x0400, 0x0118},
    		{0x4598, 0xbf1b, 0x0138},
    		{0x0008, 0x0010, 0x0139},
    		{0x0150, 0x0010, 0x0141},
    		{0x0050, 0x0010, 0x0149},
    		{0x44fd, 0x3f57, 0x0150},
    		{0x0008, 0x0008, 0x0151},
    		{0x00d0, 0x0020, 0x0159},
    		{0x0000, 0x0000, 0x0190},
    		{0x0000, 0x0000, 0x0198},
    		{0x0000, 0x0000, 0x01a0},
    		{0x0002, 0x0000, 0x01a8},
    		{0x6028, 0x3dea, 0x01d0},
    		{0x0008, 0x0010, 0x01d1},
    		{0x080b, 0x0110, 0x01d9},
    		{0x1714, 0xbe3c, 0x01e0},
    		{0x0008, 0x0010, 0x01e1},
    		{0x0000, 0x0000, 0x0220},
    		{0x0004, 0x0000, 0x0228},
    		{0x0000, 0x1000, 0x0238},
    		{0x000b, 0x0050, 0x0269},
    		{0x3000, 0x0000, 0x02b8},
    		{0x0000, 0x0003, 0x02c0},
    		{0x0000, 0x0001, 0x02c8},
    		{0x0000, 0x0000, 0x02f8},
    		{0x0001, 0x0100, 0x02f9},
    		{0x0000, 0x4100, 0x02fa},
    		{0x1004, 0x0000, 0x02fb},
    		{0xe000, 0x03ff, 0x0310},
    		{0x1fc1, 0x0100, 0x0311},
    		{0x0000, 0x0100, 0x0312},
    		{0x0000, 0x0000, 0x0313},
    	};
    
    load_cfg((void*)cin, 0x20000, 354, 0, 0);
    config(0x0, 59, 0, 0);
    execute(0x9fbf, 0, 0);
    }
    {
    /// ADORA.BlockStore %6, %alloca_6 [] : memref<2xf32> -> memref<f32>  {Id = "5", KernelName = "kernel_deriche_1"}
    store(&float_5, 0x0, 8, 0, 0);

    }
    {
    /// ADORA.BlockStore %5, %alloca [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_1"}
    store(&float_4, 0x8000, 8, 0, 0);

    }
    {
    /// ADORA.BlockStore %4, %alloca_9 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_1"}
    store(&float_8, 0xa000, 8, 0, 0);

    }
    {
    /// ADORA.BlockStore %3, %arg3 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "2", KernelName = "kernel_deriche_1"}
    uint64_t dramoffset_2 = 256 * int_15;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_3 + dramoffset_2 + roffset_2, 0x2000 + spadoffset_2, 8192, 0, 0);
    spadoffset_2 = spadoffset_2 + 8192;
    
    }
    {
    /// ADORA.BlockStore %2, %alloca_10 [] : memref<2xf32> -> memref<f32>  {Id = "1", KernelName = "kernel_deriche_1"}
    store(&float_9, 0x1a000, 8, 0, 0);

    }
  }
  
  
  
  for (int int_16 = 0; int_16 < 64; int_16 = int_16 + 32){
    {
    /// %1 = ADORA.BlockLoad %arg2 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_2"}
    uint64_t dramoffset_0 = 256 * int_16;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_2 + dramoffset_0 + roffset_0, 0x10000 + spadoffset_0, 8192, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 8192;
    
    }
    {
    /// %2 = ADORA.BlockLoad %arg3 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "1", KernelName = "kernel_deriche_2"}
    uint64_t dramoffset_1 = 256 * int_16;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_3 + dramoffset_1 + roffset_1, 0x18000 + spadoffset_1, 8192, 0, 0, 0);
    spadoffset_1 = spadoffset_1 + 8192;
    
    }
    {
    /// kernel_deriche_2
    volatile unsigned short cin[16][3] __attribute__((aligned(8))) = {
    		{0x000b, 0x0118, 0x0261},
    		{0x0000, 0x0010, 0x02a0},
    		{0x0020, 0x0000, 0x02a8},
    		{0x0000, 0x0000, 0x02b0},
    		{0x2000, 0x0000, 0x02e0},
    		{0x0041, 0x0100, 0x02e1},
    		{0x0000, 0x0100, 0x02e2},
    		{0x0000, 0x0000, 0x02e3},
    		{0x2800, 0x0000, 0x02e8},
    		{0x0041, 0x0100, 0x02e9},
    		{0x0000, 0x4100, 0x02ea},
    		{0x1004, 0x0000, 0x02eb},
    		{0x2000, 0x0000, 0x02f8},
    		{0x0041, 0x0100, 0x02f9},
    		{0x0000, 0x0100, 0x02fa},
    		{0x0000, 0x0000, 0x02fb},
    	};
    
    load_cfg((void*)cin, 0x20000, 96, 0, 0);
    config(0x0, 16, 0, 0);
    execute(0x9fbf, 0, 0);
    }
    {
    /// ADORA.BlockStore %3, %arg1 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "2", KernelName = "kernel_deriche_2"}
    uint64_t dramoffset_2 = 256 * int_16;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_1 + dramoffset_2 + roffset_2, 0x12000 + spadoffset_2, 8192, 0, 0);
    spadoffset_2 = spadoffset_2 + 8192;
    
    }
  }
  
  
  
  for (int int_17 = 0; int_17 < 64; int_17 = int_17 + 32){
    {
    /// %1 = ADORA.BlockLoad %arg1 [0, %arg4] : memref<?x64xf32> -> memref<64x32xf32>  {Id = "0", KernelName = "kernel_deriche_3"}
    uint64_t dramoffset_0 = 4 * int_17;
    uint64_t spadoffset_0 = 0;
    for(int idx_0 = 0; idx_0 < 64; idx_0++){
      uint64_t roffset_0 =  256*idx_0 ;
      load_data(arg_1 + dramoffset_0 + roffset_0, 0x10000 + spadoffset_0, 128, 0, 0, 0);
      spadoffset_0 = spadoffset_0 + 128;
    } 
    }
    {
    /// kernel_deriche_3
    volatile unsigned short cin[49][3] __attribute__((aligned(8))) = {
    		{0x0000, 0x0000, 0x0010},
    		{0x0001, 0x0100, 0x0011},
    		{0x0000, 0x5100, 0x0012},
    		{0x0004, 0x0000, 0x0013},
    		{0x0030, 0x0000, 0x0058},
    		{0x0000, 0x1000, 0x00e8},
    		{0x0000, 0x8004, 0x00f0},
    		{0x0002, 0x0000, 0x00f9},
    		{0x4598, 0xbf1b, 0x0130},
    		{0x0008, 0x0010, 0x0131},
    		{0x0050, 0x0018, 0x0139},
    		{0x0000, 0x1000, 0x0178},
    		{0x0001, 0x0000, 0x0179},
    		{0x00c0, 0x0400, 0x0180},
    		{0x0000, 0x0020, 0x0188},
    		{0x00d0, 0x0020, 0x01b9},
    		{0xb54c, 0x3de1, 0x01c0},
    		{0x0008, 0x0018, 0x01c1},
    		{0x080b, 0x0098, 0x01c9},
    		{0x000b, 0x0118, 0x01d1},
    		{0xc002, 0x3000, 0x0208},
    		{0x0080, 0x000c, 0x0210},
    		{0x0000, 0x0000, 0x0218},
    		{0x0004, 0x0000, 0x0220},
    		{0x35c4, 0xbe41, 0x0248},
    		{0x0008, 0x0100, 0x0249},
    		{0x180b, 0x0050, 0x0251},
    		{0x0050, 0x0018, 0x0259},
    		{0x44fd, 0x3f57, 0x0260},
    		{0x0008, 0x0018, 0x0261},
    		{0x0002, 0x0004, 0x0298},
    		{0x0000, 0x0000, 0x02a0},
    		{0x0200, 0x0000, 0x02a8},
    		{0x0800, 0x0004, 0x02d8},
    		{0x0841, 0x0106, 0x02d9},
    		{0x0000, 0x0100, 0x02da},
    		{0x0000, 0x0000, 0x02db},
    		{0x1000, 0x0000, 0x02e0},
    		{0x0001, 0x0100, 0x02e1},
    		{0x0000, 0x4100, 0x02e2},
    		{0x1004, 0x0000, 0x02e3},
    		{0x0800, 0x0004, 0x02e8},
    		{0x0841, 0x0106, 0x02e9},
    		{0x0000, 0x8100, 0x02ea},
    		{0x0004, 0x0000, 0x02eb},
    		{0x0000, 0x0000, 0x02f0},
    		{0x0001, 0x0100, 0x02f1},
    		{0x0000, 0x8100, 0x02f2},
    		{0x0004, 0x0000, 0x02f3},
    	};
    
    load_cfg((void*)cin, 0x20000, 294, 0, 0);
    config(0x0, 49, 0, 0);
    execute(0x9fbf, 0, 0);
    }
    {
    /// ADORA.BlockStore %5, %alloca_13 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_3"}
    store(&float_12, 0x0, 8, 0, 0);

    }
    {
    /// ADORA.BlockStore %4, %alloca_12 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_3"}
    store(&float_11, 0x12000, 8, 0, 0);

    }
    {
    /// ADORA.BlockStore %3, %alloca_11 [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_3"}
    store(&float_10, 0x14000, 8, 0, 0);

    }
    {
    /// ADORA.BlockStore %2, %arg2 [0, %arg4] : memref<64x32xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_3"}
    uint64_t dramoffset_1 = 4 * int_17;
    uint64_t spadoffset_1 = 0;
    for(int idx_0 = 0; idx_0 < 64; idx_0++){
      uint64_t roffset_1 =  256*idx_0 ;
      store(arg_2 + dramoffset_1 + roffset_1, 0x16000 + spadoffset_1, 128, 0, 0);
      spadoffset_1 = spadoffset_1 + 128;
    } 
    }
  }
  
  
  
  for (int int_18 = 0; int_18 < 64; int_18 = int_18 + 32){
    {
    /// %1 = ADORA.BlockLoad %arg1 [0, %arg4] : memref<?x64xf32> -> memref<64x32xf32>  {Id = "0", KernelName = "kernel_deriche_4"}
    uint64_t dramoffset_0 = 4 * int_18;
    uint64_t spadoffset_0 = 0;
    for(int idx_0 = 0; idx_0 < 64; idx_0++){
      uint64_t roffset_0 =  256*idx_0 ;
      load_data(arg_1 + dramoffset_0 + roffset_0, 0x10000 + spadoffset_0, 128, 0, 0, 0);
      spadoffset_0 = spadoffset_0 + 128;
    } 
    }
    {
    /// kernel_deriche_4
    volatile unsigned short cin[60][3] __attribute__((aligned(8))) = {
    		{0x0800, 0x03fc, 0x0018},
    		{0xf841, 0x0101, 0x0019},
    		{0x0000, 0x7100, 0x001a},
    		{0x1004, 0x0000, 0x001b},
    		{0x0000, 0x0000, 0x0020},
    		{0x0001, 0x0100, 0x0021},
    		{0x0000, 0x7100, 0x0022},
    		{0x0004, 0x0000, 0x0023},
    		{0x0000, 0x0000, 0x0030},
    		{0x0001, 0x0100, 0x0031},
    		{0x0000, 0x3100, 0x0032},
    		{0x0004, 0x0000, 0x0033},
    		{0x0033, 0x0000, 0x0068},
    		{0x0010, 0x0000, 0x0078},
    		{0x008b, 0x0118, 0x00b1},
    		{0x0050, 0x0018, 0x00b9},
    		{0x0000, 0x2000, 0x00e8},
    		{0x1000, 0x0000, 0x00f0},
    		{0x4040, 0x0000, 0x00f8},
    		{0x4040, 0x0000, 0x0100},
    		{0x0000, 0x0000, 0x0108},
    		{0x6028, 0x3dea, 0x0128},
    		{0x0008, 0x0020, 0x0129},
    		{0x000b, 0x0108, 0x0139},
    		{0x018b, 0x0060, 0x0141},
    		{0x44fd, 0x3f57, 0x0148},
    		{0x0008, 0x0040, 0x0149},
    		{0x0150, 0x0008, 0x0151},
    		{0x0000, 0x0000, 0x0178},
    		{0x0004, 0x0000, 0x0188},
    		{0x0000, 0x0000, 0x0190},
    		{0x0000, 0x0000, 0x0198},
    		{0x0050, 0x0018, 0x01c1},
    		{0x1714, 0xbe3c, 0x01c8},
    		{0x0008, 0x0018, 0x01c9},
    		{0x4598, 0xbf1b, 0x01d8},
    		{0x0008, 0x0010, 0x01d9},
    		{0x0040, 0x0004, 0x0208},
    		{0x0040, 0x0000, 0x0210},
    		{0x0180, 0x0010, 0x0249},
    		{0x0150, 0x0018, 0x0251},
    		{0x0010, 0x0010, 0x0298},
    		{0x0000, 0x0020, 0x02a0},
    		{0x0000, 0x0020, 0x02a8},
    		{0x0000, 0x0020, 0x02b0},
    		{0x0000, 0x0020, 0x02b8},
    		{0x0000, 0x0020, 0x02c0},
    		{0x0200, 0x0000, 0x02c8},
    		{0x0000, 0x03fc, 0x02d8},
    		{0xf841, 0x0101, 0x02d9},
    		{0x0000, 0x0100, 0x02da},
    		{0x0000, 0x0000, 0x02db},
    		{0x0800, 0x0000, 0x02e8},
    		{0x0001, 0x0100, 0x02e9},
    		{0x0000, 0x1100, 0x02ea},
    		{0x0004, 0x0000, 0x02eb},
    		{0x0000, 0x0000, 0x0310},
    		{0x0001, 0x0100, 0x0311},
    		{0x0000, 0x6100, 0x0312},
    		{0x0004, 0x0000, 0x0313},
    	};
    
    load_cfg((void*)cin, 0x20000, 360, 0, 0);
    config(0x0, 60, 0, 0);
    execute(0x9fbf, 0, 0);
    }
    {
    /// ADORA.BlockStore %6, %alloca_8 [] : memref<2xf32> -> memref<f32>  {Id = "5", KernelName = "kernel_deriche_4"}
    store(&float_7, 0x18000, 8, 0, 0);

    }
    {
    /// ADORA.BlockStore %5, %alloca_7 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_4"}
    store(&float_6, 0x12000, 8, 0, 0);

    }
    {
    /// ADORA.BlockStore %4, %alloca_6 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_4"}
    store(&float_5, 0x0, 8, 0, 0);

    }
    {
    /// ADORA.BlockStore %3, %alloca [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_4"}
    store(&float_4, 0x8000, 8, 0, 0);

    }
    {
    /// ADORA.BlockStore %2, %arg3 [0, %arg4] : memref<64x32xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_4"}
    uint64_t dramoffset_1 = 4 * int_18;
    uint64_t spadoffset_1 = 0;
    for(int idx_0 = 0; idx_0 < 64; idx_0++){
      uint64_t roffset_1 =  256*idx_0 ;
      store(arg_3 + dramoffset_1 + roffset_1, 0x2000 + spadoffset_1, 128, 0, 0);
      spadoffset_1 = spadoffset_1 + 128;
    } 
    }
  }
  
  
  
  for (int int_19 = 0; int_19 < 64; int_19 = int_19 + 32){
    {
    /// %1 = ADORA.BlockLoad %arg2 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_5"}
    uint64_t dramoffset_0 = 256 * int_19;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_2 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 8192, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 8192;
    
    }
    {
    /// %2 = ADORA.BlockLoad %arg3 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "1", KernelName = "kernel_deriche_5"}
    uint64_t dramoffset_1 = 256 * int_19;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_3 + dramoffset_1 + roffset_1, 0x10000 + spadoffset_1, 8192, 0, 0, 0);
    spadoffset_1 = spadoffset_1 + 8192;
    
    }
    {
    /// kernel_deriche_5
    volatile unsigned short cin[15][3] __attribute__((aligned(8))) = {
    		{0x000b, 0x0120, 0x0261},
    		{0x0000, 0x0000, 0x02a8},
    		{0x0002, 0x0000, 0x02b0},
    		{0x2800, 0x0000, 0x02e8},
    		{0x0041, 0x0100, 0x02e9},
    		{0x0000, 0x4100, 0x02ea},
    		{0x1004, 0x0000, 0x02eb},
    		{0x2000, 0x0000, 0x02f0},
    		{0x0041, 0x0100, 0x02f1},
    		{0x0000, 0x0100, 0x02f2},
    		{0x0000, 0x0000, 0x02f3},
    		{0x2000, 0x0000, 0x02f8},
    		{0x0041, 0x0100, 0x02f9},
    		{0x0000, 0x0100, 0x02fa},
    		{0x0000, 0x0000, 0x02fb},
    	};
    
    load_cfg((void*)cin, 0x20000, 90, 0, 0);
    config(0x0, 15, 0, 0);
    execute(0x9fbf, 0, 0);
    }
    {
    /// ADORA.BlockStore %3, %arg1 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "2", KernelName = "kernel_deriche_5"}
    uint64_t dramoffset_2 = 256 * int_19;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_1 + dramoffset_2 + roffset_2, 0x12000 + spadoffset_2, 8192, 0, 0);
    spadoffset_2 = spadoffset_2 + 8192;
    
    }
  }
  
  
  
}
