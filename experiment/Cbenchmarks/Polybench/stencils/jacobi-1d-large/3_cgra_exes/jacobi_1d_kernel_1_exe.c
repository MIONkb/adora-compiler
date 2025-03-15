
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void jacobi_1d_kernel_1(void* arg_0 ,void* arg_1){
  /// %0 = ADORA.BlockLoad %arg0 [%c0] : memref<2000xf32> -> memref<1000xf32>  {Id = "0", KernelName = "jacobi_1d_kernel_1"}
  uint64_t dramoffset_0 = 0;
  uint64_t spadoffset_0 = 0;
  uint64_t roffset_0 = 0;
  load_data(arg_0 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 4000, 0, 0, 0);
  spadoffset_0 = spadoffset_0 + 4000;
  
  int int_2 = 1;
  /// %2 = ADORA.BlockLoad %arg0 [%1] : memref<2000xf32> -> memref<1000xf32>  {Id = "1", KernelName = "jacobi_1d_kernel_1"}
  uint64_t dramoffset_1 = 4 * int_2;
  uint64_t spadoffset_1 = 0;
  uint64_t roffset_1 = 0;
  load_data(arg_0 + dramoffset_1 + roffset_1, 0x10000 + spadoffset_1, 4000, 0, 0, 0);
  spadoffset_1 = spadoffset_1 + 4000;
  
  int int_3 = 2;
  /// %4 = ADORA.BlockLoad %arg0 [%3] : memref<2000xf32> -> memref<1000xf32>  {Id = "2", KernelName = "jacobi_1d_kernel_1"}
  uint64_t dramoffset_2 = 4 * int_3;
  uint64_t spadoffset_2 = 0;
  uint64_t roffset_2 = 0;
  load_data(arg_0 + dramoffset_2 + roffset_2, 0x12000 + spadoffset_2, 4000, 0, 0, 0);
  spadoffset_2 = spadoffset_2 + 4000;
  
  /// %6 = ADORA.BlockLoad %arg0 [%c999] : memref<2000xf32> -> memref<1000xf32>  {Id = "4", KernelName = "jacobi_1d_kernel_1"}
  uint64_t dramoffset_4 = 4 * 999;
  uint64_t spadoffset_4 = 0;
  uint64_t roffset_4 = 0;
  load_data(arg_0 + dramoffset_4 + roffset_4, 0x8000 + spadoffset_4, 4000, 0, 0, 0);
  spadoffset_4 = spadoffset_4 + 4000;
  
  int int_4 = 999 + 1;
  /// %8 = ADORA.BlockLoad %arg0 [%7] : memref<2000xf32> -> memref<1000xf32>  {Id = "5", KernelName = "jacobi_1d_kernel_1"}
  uint64_t dramoffset_5 = 4 * int_4;
  uint64_t spadoffset_5 = 0;
  uint64_t roffset_5 = 0;
  load_data(arg_0 + dramoffset_5 + roffset_5, 0xa000 + spadoffset_5, 4000, 0, 0, 0);
  spadoffset_5 = spadoffset_5 + 4000;
  
  int int_5 = 999 + 2;
  /// %10 = ADORA.BlockLoad %arg0 [%9] : memref<2000xf32> -> memref<1000xf32>  {Id = "6", KernelName = "jacobi_1d_kernel_1"}
  uint64_t dramoffset_6 = 4 * int_5;
  uint64_t spadoffset_6 = 0;
  uint64_t roffset_6 = 0;
  load_data(arg_0 + dramoffset_6 + roffset_6, 0x0 + spadoffset_6, 4000, 0, 0, 0);
  spadoffset_6 = spadoffset_6 + 4000;
  
  volatile unsigned short cin[52][3] __attribute__((aligned(8))) = {
  		{0x2000, 0x9c00, 0x0010},
  		{0x000f, 0x0000, 0x0011},
  		{0x0000, 0x0100, 0x0012},
  		{0x0000, 0x0000, 0x0013},
  		{0x2800, 0x9c00, 0x0020},
  		{0x000f, 0x0000, 0x0021},
  		{0x0000, 0x9100, 0x0022},
  		{0x0200, 0x0000, 0x0023},
  		{0x2800, 0x9c00, 0x0028},
  		{0x000f, 0x0000, 0x0029},
  		{0x0000, 0x0100, 0x002a},
  		{0x0000, 0x0000, 0x002b},
  		{0x2000, 0x9c00, 0x0040},
  		{0x000f, 0x0000, 0x0041},
  		{0x0000, 0x0100, 0x0042},
  		{0x0000, 0x0000, 0x0043},
  		{0x0000, 0x0000, 0x0060},
  		{0x0000, 0x0008, 0x0068},
  		{0x1401, 0x0000, 0x0070},
  		{0x2000, 0x0000, 0x0078},
  		{0x0400, 0x0001, 0x0080},
  		{0x0000, 0x0000, 0x0088},
  		{0x0003, 0x0000, 0x00b0},
  		{0x000d, 0x0004, 0x00b1},
  		{0x020e, 0x0014, 0x00b9},
  		{0x002e, 0x0014, 0x00c9},
  		{0x0003, 0x0000, 0x01c8},
  		{0x000d, 0x0040, 0x01c9},
  		{0x0001, 0x0000, 0x0211},
  		{0x0004, 0x0004, 0x0218},
  		{0x040e, 0x0044, 0x0259},
  		{0x020e, 0x0038, 0x0261},
  		{0x0300, 0x0010, 0x02a0},
  		{0x0040, 0x0000, 0x02a8},
  		{0x0001, 0x0000, 0x02b0},
  		{0x0000, 0x0001, 0x02b8},
  		{0x2000, 0x9c00, 0x02e0},
  		{0x000f, 0x0000, 0x02e1},
  		{0x0000, 0x0100, 0x02e2},
  		{0x0000, 0x0000, 0x02e3},
  		{0x3000, 0x9c00, 0x02e8},
  		{0x000f, 0x0000, 0x02e9},
  		{0x0000, 0x9300, 0x02ea},
  		{0x0000, 0x0000, 0x02eb},
  		{0x2800, 0x9c00, 0x02f0},
  		{0x000f, 0x0000, 0x02f1},
  		{0x0000, 0x0100, 0x02f2},
  		{0x0000, 0x0000, 0x02f3},
  		{0x2000, 0x9c00, 0x0300},
  		{0x000f, 0x0000, 0x0301},
  		{0x0000, 0x0100, 0x0302},
  		{0x0000, 0x0000, 0x0303},
  	};
  
  load_cfg((void*)cin, 0x20000, 312, 0, 0);
  config(0x0, 52, 0, 0);
  execute(0x2e9a, 0, 0);
  int int_6 = 1;
  /// ADORA.BlockStore %5, %arg1 [%12] : memref<1000xf32> -> memref<2000xf32>  {Id = "3", KernelName = "jacobi_1d_kernel_1"}
  uint64_t dramoffset_3 = 4 * int_6;
  uint64_t spadoffset_3 = 0;
  uint64_t roffset_3 = 0;
  store(arg_1 + dramoffset_3 + roffset_3, 0x14000 + spadoffset_3, 4000, 0, 0);
  spadoffset_3 = spadoffset_3 + 4000;
  
  int int_7 = 999 + 1;
  /// ADORA.BlockStore %11, %arg1 [%13] : memref<1000xf32> -> memref<2000xf32>  {Id = "7", KernelName = "jacobi_1d_kernel_1"}
  uint64_t dramoffset_7 = 4 * int_7;
  uint64_t spadoffset_7 = 0;
  uint64_t roffset_7 = 0;
  store(arg_1 + dramoffset_7 + roffset_7, 0x2000 + spadoffset_7, 4000, 0, 0);
  spadoffset_7 = spadoffset_7 + 4000;

//  fence(1); 
}
