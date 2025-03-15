
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
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 8192, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 8192;
    
    }
    {
    /// kernel_deriche_0
    volatile unsigned short cin[54][3] __attribute__((aligned(8))) = {
    		{0x2000, 0x0000, 0x0040},
    		{0x0041, 0x0100, 0x0041},
    		{0x0000, 0x0100, 0x0042},
    		{0x0000, 0x0000, 0x0043},
    		{0x0000, 0x0030, 0x0058},
    		{0x0000, 0x0002, 0x0060},
    		{0x0000, 0x0002, 0x0068},
    		{0x0000, 0x0002, 0x0070},
    		{0x0000, 0x0002, 0x0078},
    		{0x0000, 0x0002, 0x0080},
    		{0x0200, 0x0000, 0x0088},
    		{0x35c4, 0xbe41, 0x00d0},
    		{0x0008, 0x0040, 0x00d1},
    		{0x0003, 0x0000, 0x00e9},
    		{0x0000, 0x000c, 0x0110},
    		{0x0000, 0x0000, 0x0118},
    		{0x180a, 0x00d0, 0x0151},
    		{0x0003, 0x0000, 0x0179},
    		{0x0000, 0x0000, 0x0198},
    		{0x0000, 0x0000, 0x01a0},
    		{0x0000, 0x0010, 0x01a8},
    		{0x4598, 0xbf1b, 0x01d0},
    		{0x0008, 0x0000, 0x01d1},
    		{0xb54c, 0x3de1, 0x01d8},
    		{0x0008, 0x0000, 0x01d9},
    		{0x180a, 0x0088, 0x01e9},
    		{0x44fd, 0x3f57, 0x01f0},
    		{0x0008, 0x0100, 0x01f1},
    		{0x0003, 0x0000, 0x0209},
    		{0x0000, 0x0000, 0x0220},
    		{0x0000, 0x4000, 0x0228},
    		{0x5000, 0x0000, 0x0230},
    		{0x0000, 0x0000, 0x0240},
    		{0x020a, 0x0048, 0x0279},
    		{0x0050, 0x0018, 0x0281},
    		{0x2000, 0x0000, 0x0298},
    		{0x0000, 0x0000, 0x02c0},
    		{0x0000, 0x0000, 0x02c8},
    		{0x0000, 0x0000, 0x02d8},
    		{0x0001, 0x0100, 0x02d9},
    		{0x0000, 0x8100, 0x02da},
    		{0x1004, 0x0000, 0x02db},
    		{0x3000, 0x0000, 0x0300},
    		{0x0041, 0x0100, 0x0301},
    		{0x0000, 0x8100, 0x0302},
    		{0x1004, 0x0000, 0x0303},
    		{0x0000, 0x0000, 0x0308},
    		{0x0001, 0x0100, 0x0309},
    		{0x0000, 0x1100, 0x030a},
    		{0x1004, 0x0000, 0x030b},
    		{0x0800, 0x0000, 0x0310},
    		{0x0001, 0x0100, 0x0311},
    		{0x0000, 0x8100, 0x0312},
    		{0x0004, 0x0000, 0x0313},
    	};
    
    load_cfg((void*)cin, 0x20000, 324, 0, 0);
    config(0x0, 54, 0, 0);
    execute(0xe180, 0, 0);
    }
    {
    /// ADORA.BlockStore %5, %alloca_14 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_0"}
    store(&float_13, 0x10000, 8, 0, 0);
    {
    /// ADORA.BlockStore %4, %alloca_11 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_0"}
    store(&float_10, 0x18000, 8, 0, 0);
    {
    /// ADORA.BlockStore %3, %alloca_12 [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_0"}
    store(&float_11, 0x1a000, 8, 0, 0);
    {
    /// ADORA.BlockStore %2, %arg2 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_0"}
    uint64_t dramoffset_1 = 256 * int_14;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    store(arg_2 + dramoffset_1 + roffset_1, 0x1c000 + spadoffset_1, 8192, 0, 0);
    spadoffset_1 = spadoffset_1 + 8192;
    
    }
  }
  
  
  
}
