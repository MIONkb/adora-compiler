
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
    /// %1 = ADORA.BlockLoad %arg0 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_0"}
    uint64_t dramoffset_0 = 256 * int_14;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x10000 + spadoffset_0, 8192, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 8192;
    
    /// kernel_deriche_0
    volatile unsigned short cin[54][3] __attribute__((aligned(8))) = {
    		{0x0800, 0x0000, 0x0038},
    		{0x0001, 0x0100, 0x0039},
    		{0x0000, 0x8100, 0x003a},
    		{0x0004, 0x0000, 0x003b},
    		{0x0000, 0x000c, 0x0068},
    		{0x0000, 0x0008, 0x0070},
    		{0x0000, 0x0008, 0x0078},
    		{0x0020, 0x0000, 0x0080},
    		{0x0000, 0x1000, 0x00f8},
    		{0x0000, 0x2000, 0x0180},
    		{0x1000, 0x1000, 0x0188},
    		{0x0000, 0x8000, 0x0190},
    		{0x1000, 0x0000, 0x0198},
    		{0xb54c, 0x3de1, 0x01b8},
    		{0x0008, 0x0100, 0x01b9},
    		{0x010a, 0x0118, 0x01c1},
    		{0x000a, 0x0108, 0x01d1},
    		{0x180a, 0x0118, 0x01d9},
    		{0x0050, 0x0008, 0x01e1},
    		{0x0004, 0x000c, 0x0208},
    		{0x0004, 0x8200, 0x0210},
    		{0x0000, 0x0800, 0x0218},
    		{0x0000, 0x0c00, 0x0220},
    		{0x0008, 0x0084, 0x0228},
    		{0x0000, 0x0000, 0x0229},
    		{0x0000, 0x0080, 0x0230},
    		{0x0050, 0x0010, 0x0249},
    		{0x35c4, 0xbe41, 0x0250},
    		{0x0008, 0x0018, 0x0251},
    		{0x44fd, 0x3f57, 0x0268},
    		{0x0008, 0x0010, 0x0269},
    		{0x0050, 0x0008, 0x0271},
    		{0x4598, 0xbf1b, 0x0278},
    		{0x0008, 0x0008, 0x0279},
    		{0x0010, 0x0010, 0x0298},
    		{0x0000, 0x0008, 0x02a0},
    		{0x2300, 0x0000, 0x02b8},
    		{0x0000, 0x0000, 0x02c0},
    		{0x2800, 0x0000, 0x02d8},
    		{0x0041, 0x0100, 0x02d9},
    		{0x0000, 0x0100, 0x02da},
    		{0x0000, 0x0000, 0x02db},
    		{0x1000, 0x0000, 0x02f8},
    		{0x0001, 0x0100, 0x02f9},
    		{0x0000, 0x8100, 0x02fa},
    		{0x1004, 0x0000, 0x02fb},
    		{0x2000, 0x0000, 0x0300},
    		{0x0041, 0x0100, 0x0301},
    		{0x0000, 0x8100, 0x0302},
    		{0x0004, 0x0000, 0x0303},
    		{0x0000, 0x0000, 0x0308},
    		{0x0001, 0x0100, 0x0309},
    		{0x0000, 0x4100, 0x030a},
    		{0x0004, 0x0000, 0x030b},
    	};
    
    load_cfg((void*)cin, 0x20000, 324, 0, 0);
    config(0x0, 54, 0, 0);
    execute(0xfbff, 0, 0);
    /// ADORA.BlockStore %5, %alloca_14 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_0"}
    store(&float_13, 0x8000, 8, 0, 0);
    /// ADORA.BlockStore %4, %alloca_11 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_0"}
    store(&float_10, 0x18000, 8, 0, 0);
    /// ADORA.BlockStore %3, %alloca_12 [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_0"}
    store(&float_11, 0x1a000, 8, 0, 0);
    /// ADORA.BlockStore %2, %arg2 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_0"}
    uint64_t dramoffset_1 = 256 * int_14;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    store(arg_2 + dramoffset_1 + roffset_1, 0x1c000 + spadoffset_1, 8192, 0, 0);
    spadoffset_1 = spadoffset_1 + 8192;
    
  }
  
  
  
  for (int int_15 = 0; int_15 < 64; int_15 = int_15 + 32){
    /// %1 = ADORA.BlockLoad %arg0 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_1"}
    uint64_t dramoffset_0 = 256 * int_15;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 8192, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 8192;
    
    /// kernel_deriche_1
    volatile unsigned short cin[58][3] __attribute__((aligned(8))) = {
    		{0xe000, 0x03ff, 0x0018},
    		{0x1fc1, 0x0100, 0x0019},
    		{0x0000, 0x7100, 0x001a},
    		{0x1004, 0x0000, 0x001b},
    		{0x1000, 0x0000, 0x0020},
    		{0x0001, 0x0100, 0x0021},
    		{0x0000, 0x3100, 0x0022},
    		{0x0004, 0x0000, 0x0023},
    		{0x0800, 0x0000, 0x0028},
    		{0x0001, 0x0100, 0x0029},
    		{0x0000, 0x7100, 0x002a},
    		{0x0004, 0x0000, 0x002b},
    		{0x1000, 0x0000, 0x0038},
    		{0x0001, 0x0100, 0x0039},
    		{0x0000, 0x2100, 0x003a},
    		{0x0004, 0x0000, 0x003b},
    		{0xe000, 0x03ff, 0x0040},
    		{0x1fc1, 0x0100, 0x0041},
    		{0x0000, 0x0100, 0x0042},
    		{0x0000, 0x0000, 0x0043},
    		{0x0000, 0x000c, 0x0058},
    		{0x1100, 0x0004, 0x0060},
    		{0x0430, 0x0000, 0x0068},
    		{0x1010, 0x0000, 0x0070},
    		{0x1030, 0x003c, 0x0080},
    		{0x0000, 0x0020, 0x0088},
    		{0x4598, 0xbf1b, 0x00a0},
    		{0x0008, 0x0010, 0x00a1},
    		{0x0050, 0x0008, 0x00a9},
    		{0x180a, 0x0050, 0x00b1},
    		{0x008a, 0x00e0, 0x00b9},
    		{0x6028, 0x3dea, 0x00c0},
    		{0x0008, 0x0010, 0x00c1},
    		{0x0050, 0x0020, 0x00c9},
    		{0x0050, 0x0018, 0x00d1},
    		{0x0000, 0x0c00, 0x00e8},
    		{0x0000, 0x0080, 0x00f0},
    		{0x0000, 0x0400, 0x00f8},
    		{0x0000, 0x0000, 0x0100},
    		{0x4000, 0x0000, 0x0108},
    		{0x0000, 0x6c04, 0x0110},
    		{0x0003, 0x0000, 0x0111},
    		{0x2080, 0x0000, 0x0118},
    		{0x0050, 0x0010, 0x0139},
    		{0x44fd, 0x3f57, 0x0140},
    		{0x0008, 0x0018, 0x0141},
    		{0x000a, 0x0050, 0x0151},
    		{0x1714, 0xbe3c, 0x0158},
    		{0x0008, 0x0010, 0x0159},
    		{0x0080, 0x0008, 0x0161},
    		{0x0000, 0x0000, 0x0188},
    		{0x0003, 0x0000, 0x01a1},
    		{0x0003, 0x0000, 0x0231},
    		{0x2000, 0x0000, 0x02c0},
    		{0x0800, 0x0000, 0x0300},
    		{0x0001, 0x0100, 0x0301},
    		{0x0000, 0x6100, 0x0302},
    		{0x1004, 0x0000, 0x0303},
    	};
    
    load_cfg((void*)cin, 0x20000, 348, 0, 0);
    config(0x0, 58, 0, 0);
    execute(0xfbff, 0, 0);
    /// ADORA.BlockStore %6, %arg3 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "5", KernelName = "kernel_deriche_1"}
    uint64_t dramoffset_5 = 256 * int_15;
    uint64_t spadoffset_5 = 0;
    uint64_t roffset_5 = 0;
    store(arg_3 + dramoffset_5 + roffset_5, 0x0 + spadoffset_5, 8192, 0, 0);
    spadoffset_5 = spadoffset_5 + 8192;
    
    /// ADORA.BlockStore %5, %alloca_9 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_1"}
    store(&float_8, 0xa000, 8, 0, 0);
    /// ADORA.BlockStore %4, %alloca_10 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_1"}
    store(&float_9, 0x18000, 8, 0, 0);
    /// ADORA.BlockStore %3, %alloca_6 [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_1"}
    store(&float_5, 0xc000, 8, 0, 0);
    /// ADORA.BlockStore %2, %alloca [] : memref<2xf32> -> memref<f32>  {Id = "1", KernelName = "kernel_deriche_1"}
    store(&float_4, 0x2000, 8, 0, 0);
  }
  
  
  
  for (int int_16 = 0; int_16 < 64; int_16 = int_16 + 32){
    /// %1 = ADORA.BlockLoad %arg2 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_2"}
    uint64_t dramoffset_0 = 256 * int_16;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_2 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 8192, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 8192;
    
    /// %2 = ADORA.BlockLoad %arg3 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "1", KernelName = "kernel_deriche_2"}
    uint64_t dramoffset_1 = 256 * int_16;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_3 + dramoffset_1 + roffset_1, 0x10000 + spadoffset_1, 8192, 0, 0, 0);
    spadoffset_1 = spadoffset_1 + 8192;
    
    /// kernel_deriche_2
    volatile unsigned short cin[19][3] __attribute__((aligned(8))) = {
    		{0x2000, 0x0000, 0x0010},
    		{0x0041, 0x0100, 0x0011},
    		{0x0000, 0x0100, 0x0012},
    		{0x0000, 0x0000, 0x0013},
    		{0x0000, 0x0000, 0x0060},
    		{0x0003, 0x0000, 0x00f1},
    		{0x0003, 0x0000, 0x0181},
    		{0x2000, 0x0000, 0x0210},
    		{0x080a, 0x0108, 0x0259},
    		{0x0000, 0x0000, 0x02a0},
    		{0x0000, 0x0000, 0x02a8},
    		{0x2800, 0x0000, 0x02e0},
    		{0x0041, 0x0100, 0x02e1},
    		{0x0000, 0x5100, 0x02e2},
    		{0x1004, 0x0000, 0x02e3},
    		{0x2000, 0x0000, 0x02f0},
    		{0x0041, 0x0100, 0x02f1},
    		{0x0000, 0x0100, 0x02f2},
    		{0x0000, 0x0000, 0x02f3},
    	};
    
    load_cfg((void*)cin, 0x20000, 114, 0, 0);
    config(0x0, 19, 0, 0);
    execute(0xfbff, 0, 0);
    /// ADORA.BlockStore %3, %arg1 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "2", KernelName = "kernel_deriche_2"}
    uint64_t dramoffset_2 = 256 * int_16;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_1 + dramoffset_2 + roffset_2, 0x12000 + spadoffset_2, 8192, 0, 0);
    spadoffset_2 = spadoffset_2 + 8192;
    
  }
  
  
  
  for (int int_17 = 0; int_17 < 64; int_17 = int_17 + 32){
    /// %1 = ADORA.BlockLoad %arg1 [0, %arg4] : memref<?x64xf32> -> memref<64x32xf32>  {Id = "0", KernelName = "kernel_deriche_3"}
    uint64_t dramoffset_0 = 4 * int_17;
    uint64_t spadoffset_0 = 0;
    for(int idx_0 = 0; idx_0 < 64; idx_0++){
      uint64_t roffset_0 =  256*idx_0 ;
      load_data(arg_1 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 128, 0, 0, 0);
      spadoffset_0 = spadoffset_0 + 128;
    } 
    /// kernel_deriche_3
    volatile unsigned short cin[45][3] __attribute__((aligned(8))) = {
    		{0x1000, 0x0000, 0x0030},
    		{0x0001, 0x0100, 0x0031},
    		{0x0000, 0x8100, 0x0032},
    		{0x1004, 0x0000, 0x0033},
    		{0x0000, 0x0000, 0x0038},
    		{0x0001, 0x0100, 0x0039},
    		{0x0000, 0x4100, 0x003a},
    		{0x1004, 0x0000, 0x003b},
    		{0x0000, 0x0004, 0x0040},
    		{0x0841, 0x0106, 0x0041},
    		{0x0000, 0x8100, 0x0042},
    		{0x0004, 0x0000, 0x0043},
    		{0x2000, 0x0000, 0x0080},
    		{0x0010, 0x0001, 0x0088},
    		{0x0050, 0x0010, 0x00c1},
    		{0x020a, 0x0118, 0x00c9},
    		{0x0050, 0x0018, 0x00d1},
    		{0x4000, 0x0000, 0x0108},
    		{0x0040, 0x0000, 0x0110},
    		{0x4004, 0x0004, 0x0118},
    		{0x4598, 0xbf1b, 0x0150},
    		{0x0008, 0x0040, 0x0151},
    		{0x000a, 0x0110, 0x0159},
    		{0x44fd, 0x3f57, 0x0160},
    		{0x0008, 0x0040, 0x0161},
    		{0x0004, 0x0010, 0x01a8},
    		{0x100a, 0x0098, 0x01e9},
    		{0xb54c, 0x3de1, 0x01f0},
    		{0x0008, 0x0018, 0x01f1},
    		{0x0000, 0x8000, 0x0228},
    		{0x1040, 0x0000, 0x0230},
    		{0x0040, 0x0000, 0x0238},
    		{0x35c4, 0xbe41, 0x0270},
    		{0x0008, 0x0020, 0x0271},
    		{0x00d0, 0x0008, 0x0279},
    		{0x0000, 0x000c, 0x02b8},
    		{0x1000, 0x0001, 0x02c0},
    		{0x0800, 0x0000, 0x0300},
    		{0x0001, 0x0100, 0x0301},
    		{0x0000, 0x3100, 0x0302},
    		{0x1004, 0x0000, 0x0303},
    		{0x0800, 0x0004, 0x0308},
    		{0x0841, 0x0106, 0x0309},
    		{0x0000, 0x0100, 0x030a},
    		{0x0000, 0x0000, 0x030b},
    	};
    
    load_cfg((void*)cin, 0x20000, 270, 0, 0);
    config(0x0, 45, 0, 0);
    execute(0xfbff, 0, 0);
    /// ADORA.BlockStore %5, %alloca_11 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_3"}
    store(&float_10, 0x8000, 8, 0, 0);
    /// ADORA.BlockStore %4, %alloca_13 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_3"}
    store(&float_12, 0x1a000, 8, 0, 0);
    /// ADORA.BlockStore %3, %alloca_12 [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_3"}
    store(&float_11, 0xa000, 8, 0, 0);
    /// ADORA.BlockStore %2, %arg2 [0, %arg4] : memref<64x32xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_3"}
    uint64_t dramoffset_1 = 4 * int_17;
    uint64_t spadoffset_1 = 0;
    for(int idx_0 = 0; idx_0 < 64; idx_0++){
      uint64_t roffset_1 =  256*idx_0 ;
      store(arg_2 + dramoffset_1 + roffset_1, 0xc000 + spadoffset_1, 8192, 0, 0);
      spadoffset_1 = spadoffset_1 + 8192;
    } 
  }
  
  
  
  for (int int_18 = 0; int_18 < 64; int_18 = int_18 + 32){
    /// %1 = ADORA.BlockLoad %arg1 [0, %arg4] : memref<?x64xf32> -> memref<64x32xf32>  {Id = "0", KernelName = "kernel_deriche_4"}
    uint64_t dramoffset_0 = 4 * int_18;
    uint64_t spadoffset_0 = 0;
    for(int idx_0 = 0; idx_0 < 64; idx_0++){
      uint64_t roffset_0 =  256*idx_0 ;
      load_data(arg_1 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 128, 0, 0, 0);
      spadoffset_0 = spadoffset_0 + 128;
    } 
    /// kernel_deriche_4
    volatile unsigned short cin[59][3] __attribute__((aligned(8))) = {
    		{0x0800, 0x0000, 0x0008},
    		{0x0001, 0x0100, 0x0009},
    		{0x0000, 0x7100, 0x000a},
    		{0x1004, 0x0000, 0x000b},
    		{0x0000, 0x03fc, 0x0010},
    		{0xf841, 0x0101, 0x0011},
    		{0x0000, 0x7100, 0x0012},
    		{0x0004, 0x0000, 0x0013},
    		{0x1000, 0x0000, 0x0020},
    		{0x0001, 0x0100, 0x0021},
    		{0x0000, 0x3100, 0x0022},
    		{0x0004, 0x0000, 0x0023},
    		{0x1033, 0x0000, 0x0058},
    		{0x0030, 0x0000, 0x0068},
    		{0x4598, 0xbf1b, 0x0098},
    		{0x0008, 0x0010, 0x0099},
    		{0x0150, 0x0020, 0x00a1},
    		{0x008a, 0x0120, 0x00b1},
    		{0x0000, 0x0000, 0x00e8},
    		{0x0000, 0x000c, 0x00f0},
    		{0x0000, 0x0400, 0x00f8},
    		{0x0006, 0x0000, 0x0100},
    		{0x180a, 0x0050, 0x0131},
    		{0x0050, 0x0018, 0x0139},
    		{0x44fd, 0x3f57, 0x0140},
    		{0x0008, 0x0018, 0x0141},
    		{0x0000, 0x0000, 0x0180},
    		{0x0000, 0x0004, 0x0188},
    		{0x0000, 0x0000, 0x0190},
    		{0x0000, 0x000c, 0x0198},
    		{0x0000, 0x0080, 0x01a0},
    		{0x0050, 0x0010, 0x01c9},
    		{0x0100, 0x0020, 0x01d1},
    		{0x000a, 0x00d0, 0x01d9},
    		{0x1714, 0xbe3c, 0x01e8},
    		{0x0008, 0x0020, 0x01e9},
    		{0x4000, 0x0000, 0x0210},
    		{0x0000, 0x2000, 0x0218},
    		{0x0101, 0x0000, 0x0220},
    		{0x0000, 0x0180, 0x0228},
    		{0x0000, 0x8180, 0x0230},
    		{0x1000, 0x0080, 0x0238},
    		{0x6028, 0x3dea, 0x0258},
    		{0x0008, 0x0040, 0x0259},
    		{0x0050, 0x0008, 0x0281},
    		{0x3000, 0x000c, 0x02c0},
    		{0x0000, 0x0001, 0x02c8},
    		{0x0800, 0x0000, 0x0300},
    		{0x0001, 0x0100, 0x0301},
    		{0x0000, 0x4100, 0x0302},
    		{0x1004, 0x0000, 0x0303},
    		{0x1000, 0x0000, 0x0308},
    		{0x0001, 0x0100, 0x0309},
    		{0x0000, 0x1100, 0x030a},
    		{0x1004, 0x0000, 0x030b},
    		{0x0000, 0x03fc, 0x0310},
    		{0xf841, 0x0101, 0x0311},
    		{0x0000, 0x0100, 0x0312},
    		{0x0000, 0x0000, 0x0313},
    	};
    
    load_cfg((void*)cin, 0x20000, 354, 0, 0);
    config(0x0, 59, 0, 0);
    execute(0xfbff, 0, 0);
    /// ADORA.BlockStore %6, %alloca_8 [] : memref<2xf32> -> memref<f32>  {Id = "5", KernelName = "kernel_deriche_4"}
    store(&float_7, 0x1a000, 8, 0, 0);
    /// ADORA.BlockStore %5, %alloca_7 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_4"}
    store(&float_6, 0x1c000, 8, 0, 0);
    /// ADORA.BlockStore %4, %arg3 [0, %arg4] : memref<64x32xf32> -> memref<?x64xf32>  {Id = "3", KernelName = "kernel_deriche_4"}
    uint64_t dramoffset_3 = 4 * int_18;
    uint64_t spadoffset_3 = 0;
    for(int idx_0 = 0; idx_0 < 64; idx_0++){
      uint64_t roffset_3 =  256*idx_0 ;
      store(arg_3 + dramoffset_3 + roffset_3, 0x0 + spadoffset_3, 8192, 0, 0);
      spadoffset_3 = spadoffset_3 + 8192;
    } 
    /// ADORA.BlockStore %3, %alloca_6 [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_4"}
    store(&float_5, 0x2000, 8, 0, 0);
    /// ADORA.BlockStore %2, %alloca [] : memref<2xf32> -> memref<f32>  {Id = "1", KernelName = "kernel_deriche_4"}
    store(&float_4, 0x4000, 8, 0, 0);
  }
  
  
  
  for (int int_19 = 0; int_19 < 64; int_19 = int_19 + 32){
    /// %1 = ADORA.BlockLoad %arg2 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_5"}
    uint64_t dramoffset_0 = 256 * int_19;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_2 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 8192, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 8192;
    
    /// %2 = ADORA.BlockLoad %arg3 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "1", KernelName = "kernel_deriche_5"}
    uint64_t dramoffset_1 = 256 * int_19;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_3 + dramoffset_1 + roffset_1, 0x18000 + spadoffset_1, 8192, 0, 0, 0);
    spadoffset_1 = spadoffset_1 + 8192;
    
    /// kernel_deriche_5
    volatile unsigned short cin[18][3] __attribute__((aligned(8))) = {
    		{0x2000, 0x0000, 0x0038},
    		{0x0041, 0x0100, 0x0039},
    		{0x0000, 0x0100, 0x003a},
    		{0x0000, 0x0000, 0x003b},
    		{0x0000, 0x0010, 0x0080},
    		{0x0003, 0x0000, 0x0111},
    		{0x0003, 0x0000, 0x01a1},
    		{0x0000, 0x0008, 0x0230},
    		{0x080a, 0x0110, 0x0271},
    		{0x0002, 0x0000, 0x02c0},
    		{0x2000, 0x0000, 0x0300},
    		{0x0041, 0x0100, 0x0301},
    		{0x0000, 0x0100, 0x0302},
    		{0x0000, 0x0000, 0x0303},
    		{0x2800, 0x0000, 0x0308},
    		{0x0041, 0x0100, 0x0309},
    		{0x0000, 0x5100, 0x030a},
    		{0x0004, 0x0000, 0x030b},
    	};
    
    load_cfg((void*)cin, 0x20000, 108, 0, 0);
    config(0x0, 18, 0, 0);
    execute(0xfbff, 0, 0);
    /// ADORA.BlockStore %3, %arg1 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "2", KernelName = "kernel_deriche_5"}
    uint64_t dramoffset_2 = 256 * int_19;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_1 + dramoffset_2 + roffset_2, 0x1a000 + spadoffset_2, 8192, 0, 0);
    spadoffset_2 = spadoffset_2 + 8192;
    
  }
  
  
  
}
