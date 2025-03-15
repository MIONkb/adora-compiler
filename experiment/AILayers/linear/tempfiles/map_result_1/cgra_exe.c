
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void forward_kernel_1(void* arg_0 ,void* arg_1 ,void* arg_2){
  /// %0 = ADORA.BlockLoad %arg0 [%c0, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "0", KernelName = "forward_kernel_1"}
  uint64_t dramoffset_0 = 0;
  uint64_t spadoffset_0 = 0;
  uint64_t roffset_0 = 0;
  load_data(arg_0 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 4096, 0, 0, 0);
  spadoffset_0 = spadoffset_0 + 4096;
  
  /// %1 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "1", KernelName = "forward_kernel_1"}
  uint64_t dramoffset_1 = 0;
  uint64_t spadoffset_1 = 0;
  uint64_t roffset_1 = 0;
  load_data(arg_1 + dramoffset_1 + roffset_1, 0xa000 + spadoffset_1, 256, 0, 0, 0);
  spadoffset_1 = spadoffset_1 + 256;
  
  /// %3 = ADORA.BlockLoad %arg0 [%c16, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "3", KernelName = "forward_kernel_1"}
  uint64_t dramoffset_3 = 256 * 16;
  uint64_t spadoffset_3 = 0;
  uint64_t roffset_3 = 0;
  load_data(arg_0 + dramoffset_3 + roffset_3, 0x18000 + spadoffset_3, 4096, 0, 0, 0);
  spadoffset_3 = spadoffset_3 + 4096;
  
  /// %4 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "4", KernelName = "forward_kernel_1"}
  uint64_t dramoffset_4 = 0;
  uint64_t spadoffset_4 = 0;
  uint64_t roffset_4 = 0;
  load_data(arg_1 + dramoffset_4 + roffset_4, 0x1a000 + spadoffset_4, 256, 0, 0, 0);
  spadoffset_4 = spadoffset_4 + 256;
  
  /// %6 = ADORA.BlockLoad %arg0 [%c32, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "6", KernelName = "forward_kernel_1"}
  uint64_t dramoffset_6 = 256 * 32;
  uint64_t spadoffset_6 = 0;
  uint64_t roffset_6 = 0;
  load_data(arg_0 + dramoffset_6 + roffset_6, 0xc000 + spadoffset_6, 4096, 0, 0, 0);
  spadoffset_6 = spadoffset_6 + 4096;
  
  /// %7 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "7", KernelName = "forward_kernel_1"}
  uint64_t dramoffset_7 = 0;
  uint64_t spadoffset_7 = 0;
  uint64_t roffset_7 = 0;
  load_data(arg_1 + dramoffset_7 + roffset_7, 0x0 + spadoffset_7, 256, 0, 0, 0);
  spadoffset_7 = spadoffset_7 + 256;
  
  /// %9 = ADORA.BlockLoad %arg0 [%c48, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "9", KernelName = "forward_kernel_1"}
  uint64_t dramoffset_9 = 256 * 48;
  uint64_t spadoffset_9 = 0;
  uint64_t roffset_9 = 0;
  load_data(arg_0 + dramoffset_9 + roffset_9, 0x10000 + spadoffset_9, 4096, 0, 0, 0);
  spadoffset_9 = spadoffset_9 + 4096;
  
  /// %10 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "10", KernelName = "forward_kernel_1"}
  uint64_t dramoffset_10 = 0;
  uint64_t spadoffset_10 = 0;
  uint64_t roffset_10 = 0;
  load_data(arg_1 + dramoffset_10 + roffset_10, 0x12000 + spadoffset_10, 256, 0, 0, 0);
  spadoffset_10 = spadoffset_10 + 256;
  
  volatile unsigned short cin[60][3] __attribute__((aligned(8))) = {
  		{0x2000, 0x0000, 0x0018},
  		{0xf041, 0x0087, 0x0019},
  		{0x0000, 0x0100, 0x001a},
  		{0x0000, 0x0000, 0x001b},
  		{0x2800, 0x0000, 0x0020},
  		{0x0041, 0x0080, 0x0021},
  		{0x0000, 0x8900, 0x0022},
  		{0x0200, 0x0000, 0x0023},
  		{0x3000, 0x0000, 0x0028},
  		{0x0041, 0x0080, 0x0029},
  		{0x0000, 0x0100, 0x002a},
  		{0x0000, 0x0000, 0x002b},
  		{0x2000, 0x0000, 0x0030},
  		{0x0041, 0x0080, 0x0031},
  		{0x0000, 0x0100, 0x0032},
  		{0x0000, 0x0000, 0x0033},
  		{0x3800, 0x0000, 0x0038},
  		{0x0041, 0x0080, 0x0039},
  		{0x0000, 0x8900, 0x003a},
  		{0x0200, 0x0000, 0x003b},
  		{0x2800, 0x0000, 0x0040},
  		{0xf041, 0x0087, 0x0041},
  		{0x0000, 0x0100, 0x0042},
  		{0x0000, 0x0000, 0x0043},
  		{0x0000, 0x0000, 0x0068},
  		{0x0001, 0x0000, 0x0070},
  		{0x0000, 0x0000, 0x0080},
  		{0x0001, 0x0000, 0x0088},
  		{0x000e, 0x0014, 0x00b1},
  		{0x000e, 0x0022, 0x00c9},
  		{0x000e, 0x0046, 0x0251},
  		{0x000e, 0x0046, 0x0271},
  		{0x0010, 0x0000, 0x0298},
  		{0x0002, 0x0000, 0x02a0},
  		{0x0110, 0x0000, 0x02b8},
  		{0x0000, 0x0000, 0x02c0},
  		{0x2000, 0x0000, 0x02d8},
  		{0x0041, 0x0080, 0x02d9},
  		{0x0000, 0x0100, 0x02da},
  		{0x0000, 0x0000, 0x02db},
  		{0x2800, 0x0000, 0x02e0},
  		{0xf041, 0x0087, 0x02e1},
  		{0x0000, 0x0100, 0x02e2},
  		{0x0000, 0x0000, 0x02e3},
  		{0x3000, 0x0000, 0x02e8},
  		{0x0041, 0x0080, 0x02e9},
  		{0x0000, 0x8900, 0x02ea},
  		{0x0000, 0x0000, 0x02eb},
  		{0x2000, 0x0000, 0x02f8},
  		{0x0041, 0x0080, 0x02f9},
  		{0x0000, 0x0100, 0x02fa},
  		{0x0000, 0x0000, 0x02fb},
  		{0x3000, 0x0000, 0x0300},
  		{0x0041, 0x0080, 0x0301},
  		{0x0000, 0x8900, 0x0302},
  		{0x0000, 0x0000, 0x0303},
  		{0x2800, 0x0000, 0x0308},
  		{0xf041, 0x0087, 0x0309},
  		{0x0000, 0x0100, 0x030a},
  		{0x0000, 0x0000, 0x030b},
  	};
  
  load_cfg((void*)cin, 0x20000, 360, 0, 0);
  config(0x0, 60, 0, 0);
  execute(0x77fc, 0, 0);
  /// ADORA.BlockStore %2, %arg2 [%c0, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
  uint64_t dramoffset_2 = 0;
  uint64_t spadoffset_2 = 0;
  uint64_t roffset_2 = 0;
  store(arg_2 + dramoffset_2 + roffset_2, 0xe000 + spadoffset_2, 4096, 0, 0);
  spadoffset_2 = spadoffset_2 + 4096;
  
  /// ADORA.BlockStore %5, %arg2 [%c16, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "5", KernelName = "forward_kernel_1"}
  uint64_t dramoffset_5 = 256 * 16;
  uint64_t spadoffset_5 = 0;
  uint64_t roffset_5 = 0;
  store(arg_2 + dramoffset_5 + roffset_5, 0x1c000 + spadoffset_5, 4096, 0, 0);
  spadoffset_5 = spadoffset_5 + 4096;
  
  /// ADORA.BlockStore %8, %arg2 [%c32, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "8", KernelName = "forward_kernel_1"}
  uint64_t dramoffset_8 = 256 * 32;
  uint64_t spadoffset_8 = 0;
  uint64_t roffset_8 = 0;
  store(arg_2 + dramoffset_8 + roffset_8, 0x2000 + spadoffset_8, 4096, 0, 0);
  spadoffset_8 = spadoffset_8 + 4096;
  
  /// ADORA.BlockStore %11, %arg2 [%c48, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "11", KernelName = "forward_kernel_1"}
  uint64_t dramoffset_11 = 256 * 48;
  uint64_t spadoffset_11 = 0;
  uint64_t roffset_11 = 0;
  store(arg_2 + dramoffset_11 + roffset_11, 0x14000 + spadoffset_11, 4096, 0, 0);
  spadoffset_11 = spadoffset_11 + 4096;
  
}
