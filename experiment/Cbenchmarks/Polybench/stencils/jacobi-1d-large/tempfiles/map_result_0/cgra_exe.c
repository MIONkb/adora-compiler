
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void jacobi_1d_kernel_0(void* arg_0 ,void* arg_1){
  /// %0 = ADORA.BlockLoad %arg0 [%c0] : memref<2000xf32> -> memref<1000xf32>  {Id = "0", KernelName = "jacobi_1d_kernel_0"}
  uint64_t dramoffset_0 = 0;
  uint64_t spadoffset_0 = 0;
  uint64_t roffset_0 = 0;
  load_data(arg_0 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 4000, 0, 0, 0);
  spadoffset_0 = spadoffset_0 + 4000;
  
  int int_2 = 1;
  /// %2 = ADORA.BlockLoad %arg0 [%1] : memref<2000xf32> -> memref<1000xf32>  {Id = "1", KernelName = "jacobi_1d_kernel_0"}
  uint64_t dramoffset_1 = 4 * int_2;
  uint64_t spadoffset_1 = 0;
  uint64_t roffset_1 = 0;
  load_data(arg_0 + dramoffset_1 + roffset_1, 0x2000 + spadoffset_1, 4000, 0, 0, 0);
  spadoffset_1 = spadoffset_1 + 4000;
  
  int int_3 = 2;
  /// %4 = ADORA.BlockLoad %arg0 [%3] : memref<2000xf32> -> memref<1000xf32>  {Id = "2", KernelName = "jacobi_1d_kernel_0"}
  uint64_t dramoffset_2 = 4 * int_3;
  uint64_t spadoffset_2 = 0;
  uint64_t roffset_2 = 0;
  load_data(arg_0 + dramoffset_2 + roffset_2, 0x18000 + spadoffset_2, 4000, 0, 0, 0);
  spadoffset_2 = spadoffset_2 + 4000;
  
  /// %6 = ADORA.BlockLoad %arg0 [%c999] : memref<2000xf32> -> memref<1000xf32>  {Id = "4", KernelName = "jacobi_1d_kernel_0"}
  uint64_t dramoffset_4 = 4 * 999;
  uint64_t spadoffset_4 = 0;
  uint64_t roffset_4 = 0;
  load_data(arg_0 + dramoffset_4 + roffset_4, 0x10000 + spadoffset_4, 4000, 0, 0, 0);
  spadoffset_4 = spadoffset_4 + 4000;
  
  int int_4 = 999 + 1;
  /// %8 = ADORA.BlockLoad %arg0 [%7] : memref<2000xf32> -> memref<1000xf32>  {Id = "5", KernelName = "jacobi_1d_kernel_0"}
  uint64_t dramoffset_5 = 4 * int_4;
  uint64_t spadoffset_5 = 0;
  uint64_t roffset_5 = 0;
  load_data(arg_0 + dramoffset_5 + roffset_5, 0x4000 + spadoffset_5, 4000, 0, 0, 0);
  spadoffset_5 = spadoffset_5 + 4000;
  
  int int_5 = 999 + 2;
  /// %10 = ADORA.BlockLoad %arg0 [%9] : memref<2000xf32> -> memref<1000xf32>  {Id = "6", KernelName = "jacobi_1d_kernel_0"}
  uint64_t dramoffset_6 = 4 * int_5;
  uint64_t spadoffset_6 = 0;
  uint64_t roffset_6 = 0;
  load_data(arg_0 + dramoffset_6 + roffset_6, 0x8000 + spadoffset_6, 4000, 0, 0, 0);
  spadoffset_6 = spadoffset_6 + 4000;
  
  volatile unsigned short cin[58][3] __attribute__((aligned(8))) = {
  		{0x3000, 0x9c00, 0x0008},
  		{0x000f, 0x0000, 0x0009},
  		{0x0000, 0x0100, 0x000a},
  		{0x0000, 0x0000, 0x000b},
  		{0x2800, 0x9c00, 0x0010},
  		{0x000f, 0x0000, 0x0011},
  		{0x0000, 0x0100, 0x0012},
  		{0x0000, 0x0000, 0x0013},
  		{0x2000, 0x9c00, 0x0020},
  		{0x000f, 0x0000, 0x0021},
  		{0x0000, 0x0100, 0x0022},
  		{0x0000, 0x0000, 0x0023},
  		{0x3000, 0x9c00, 0x0028},
  		{0x000f, 0x0000, 0x0029},
  		{0x0000, 0x9100, 0x002a},
  		{0x0200, 0x0000, 0x002b},
  		{0x2800, 0x9c00, 0x0030},
  		{0x000f, 0x0000, 0x0031},
  		{0x0000, 0x9100, 0x0032},
  		{0x0200, 0x0000, 0x0033},
  		{0x2000, 0x9c00, 0x0038},
  		{0x000f, 0x0000, 0x0039},
  		{0x0000, 0x0100, 0x003a},
  		{0x0000, 0x0000, 0x003b},
  		{0x0000, 0x0000, 0x0058},
  		{0x0000, 0x0004, 0x0060},
  		{0x0100, 0x0000, 0x0068},
  		{0x8000, 0x0000, 0x0070},
  		{0x0001, 0x0002, 0x0078},
  		{0x0001, 0x0000, 0x0080},
  		{0x020e, 0x0018, 0x00a1},
  		{0x000e, 0x0014, 0x00a9},
  		{0x020e, 0x0022, 0x00b1},
  		{0x0003, 0x0000, 0x00b8},
  		{0x000d, 0x0006, 0x00b9},
  		{0x0003, 0x0000, 0x00c0},
  		{0x000d, 0x0006, 0x00c1},
  		{0x0002, 0x0000, 0x00f0},
  		{0x0000, 0x0000, 0x00f8},
  		{0x1000, 0x0000, 0x0100},
  		{0x0040, 0x0000, 0x0108},
  		{0x002e, 0x0042, 0x0149},
  		{0x0000, 0x1000, 0x0180},
  		{0x0008, 0x0000, 0x0198},
  		{0x0000, 0x0180, 0x01a0},
  		{0x0000, 0x0200, 0x01a8},
  		{0x0000, 0x1000, 0x0210},
  		{0x0000, 0x1000, 0x0238},
  		{0x0000, 0x0000, 0x02a0},
  		{0x0000, 0x0000, 0x02c8},
  		{0x2000, 0x9c00, 0x02e8},
  		{0x000f, 0x0000, 0x02e9},
  		{0x0000, 0x0100, 0x02ea},
  		{0x0000, 0x0000, 0x02eb},
  		{0x2000, 0x9c00, 0x0310},
  		{0x000f, 0x0000, 0x0311},
  		{0x0000, 0x0100, 0x0312},
  		{0x0000, 0x0000, 0x0313},
  	};
  
  load_cfg((void*)cin, 0x20000, 348, 0, 0);
  config(0x0, 58, 0, 0);
  execute(0x847b, 0, 0);
  int int_6 = 1;
  /// ADORA.BlockStore %5, %arg1 [%12] : memref<1000xf32> -> memref<2000xf32>  {Id = "3", KernelName = "jacobi_1d_kernel_0"}
  uint64_t dramoffset_3 = 4 * int_6;
  uint64_t spadoffset_3 = 0;
  uint64_t roffset_3 = 0;
  store(arg_1 + dramoffset_3 + roffset_3, 0xa000 + spadoffset_3, 4000, 0, 0);
  spadoffset_3 = spadoffset_3 + 4000;
  
  int int_7 = 999 + 1;
  /// ADORA.BlockStore %11, %arg1 [%13] : memref<1000xf32> -> memref<2000xf32>  {Id = "7", KernelName = "jacobi_1d_kernel_0"}
  uint64_t dramoffset_7 = 4 * int_7;
  uint64_t spadoffset_7 = 0;
  uint64_t roffset_7 = 0;
  store(arg_1 + dramoffset_7 + roffset_7, 0xc000 + spadoffset_7, 4000, 0, 0);
  spadoffset_7 = spadoffset_7 + 4000;
  
}
