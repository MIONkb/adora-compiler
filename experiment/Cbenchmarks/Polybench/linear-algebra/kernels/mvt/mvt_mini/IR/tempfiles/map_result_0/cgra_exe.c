
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void kernel_mvt(void* arg_0 ,void* arg_1 ,void* arg_2 ,void* arg_3 ,void* arg_4){
  {
  /// %0 = ADORA.BlockLoad %arg0 [0] : memref<?xf32> -> memref<40xf32>  {Id = "0", KernelName = "kernel_mvt_0"}
  uint64_t dramoffset_0 = 0;
  uint64_t spadoffset_0 = 0;
  uint64_t roffset_0 = 0;
  load_data(arg_0 + dramoffset_0 + roffset_0, 0x10000 + spadoffset_0, 160, 0, 0, 0);
  spadoffset_0 = spadoffset_0 + 160;
  
  }
  {
  /// %1 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x40xf32> -> memref<40x40xf32>  {Id = "1", KernelName = "kernel_mvt_0"}
  uint64_t dramoffset_1 = 0;
  uint64_t spadoffset_1 = 0;
  uint64_t roffset_1 = 0;
  load_data(arg_4 + dramoffset_1 + roffset_1, 0x18000 + spadoffset_1, 6400, 0, 0, 0);
  spadoffset_1 = spadoffset_1 + 6400;
  
  }
  {
  /// %2 = ADORA.BlockLoad %arg2 [0] : memref<?xf32> -> memref<40xf32>  {Id = "2", KernelName = "kernel_mvt_0"}
  uint64_t dramoffset_2 = 0;
  uint64_t spadoffset_2 = 0;
  uint64_t roffset_2 = 0;
  load_data(arg_2 + dramoffset_2 + roffset_2, 0x0 + spadoffset_2, 160, 0, 0, 0);
  spadoffset_2 = spadoffset_2 + 160;
  
  }
  {
  /// kernel_mvt_0
  volatile unsigned short cin[32][3] __attribute__((aligned(8))) = {
  		{0x2800, 0xa000, 0x0018},
  		{0xf640, 0x0147, 0x0019},
  		{0x0000, 0x0100, 0x001a},
  		{0x0000, 0x0000, 0x001b},
  		{0x0000, 0x0000, 0x0068},
  		{0x0000, 0x6000, 0x00f8},
  		{0x8000, 0x0000, 0x0100},
  		{0x0008, 0x0060, 0x0149},
  		{0x0001, 0x0000, 0x0198},
  		{0x0000, 0x0200, 0x01a0},
  		{0x001a, 0x0008, 0x01e1},
  		{0x0000, 0x1000, 0x01e2},
  		{0x0600, 0x040a, 0x01e3},
  		{0x0000, 0x0000, 0x01e4},
  		{0x0000, 0x0000, 0x0228},
  		{0x0000, 0x1000, 0x0230},
  		{0x200a, 0x00d0, 0x0269},
  		{0x0000, 0x0000, 0x02b0},
  		{0x0000, 0x0000, 0x02b8},
  		{0x0000, 0x0000, 0x02c0},
  		{0x0000, 0xa000, 0x02f0},
  		{0x0040, 0x0140, 0x02f1},
  		{0x0000, 0x0100, 0x02f2},
  		{0x0000, 0x0000, 0x02f3},
  		{0x1000, 0xa000, 0x0300},
  		{0x0040, 0x0140, 0x0301},
  		{0x0000, 0x8100, 0x0302},
  		{0x0004, 0x0000, 0x0303},
  		{0x2000, 0xa000, 0x0308},
  		{0x0040, 0x0140, 0x0309},
  		{0x0000, 0x0100, 0x030a},
  		{0x0000, 0x0000, 0x030b},
  	};
  
  load_cfg((void*)cin, 0x20000, 192, 0, 0);
  config(0x0, 32, 0, 0);
  execute(0x7c04, 0, 0);
  }
  {
  /// ADORA.BlockStore %3, %arg0 [0] : memref<40xf32> -> memref<?xf32>  {Id = "3", KernelName = "kernel_mvt_0"}
  uint64_t dramoffset_3 = 0;
  uint64_t spadoffset_3 = 0;
  uint64_t roffset_3 = 0;
  store(arg_0 + dramoffset_3 + roffset_3, 0x1a000 + spadoffset_3, 160, 0, 0);
  spadoffset_3 = spadoffset_3 + 160;
  
  }
  {
  /// %4 = ADORA.BlockLoad %arg1 [0] : memref<?xf32> -> memref<40xf32>  {Id = "0", KernelName = "kernel_mvt_1"}
  uint64_t dramoffset_0 = 0;
  uint64_t spadoffset_0 = 0;
  uint64_t roffset_0 = 0;
  load_data(arg_1 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 160, 0, 0, 0);
  spadoffset_0 = spadoffset_0 + 160;
  
  }
  {
  /// %5 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x40xf32> -> memref<40x40xf32>  {Id = "1", KernelName = "kernel_mvt_1"}
  uint64_t dramoffset_1 = 0;
  uint64_t spadoffset_1 = 0;
  uint64_t roffset_1 = 0;
  load_data(arg_4 + dramoffset_1 + roffset_1, 0x10000 + spadoffset_1, 6400, 0, 0, 0);
  spadoffset_1 = spadoffset_1 + 6400;
  
  }
  {
  /// %6 = ADORA.BlockLoad %arg3 [0] : memref<?xf32> -> memref<40xf32>  {Id = "2", KernelName = "kernel_mvt_1"}
  uint64_t dramoffset_2 = 0;
  uint64_t spadoffset_2 = 0;
  uint64_t roffset_2 = 0;
  load_data(arg_3 + dramoffset_2 + roffset_2, 0x1a000 + spadoffset_2, 160, 0, 0, 0);
  spadoffset_2 = spadoffset_2 + 160;
  
  }
  {
  /// kernel_mvt_1
  volatile unsigned short cin[28][3] __attribute__((aligned(8))) = {
  		{0x001a, 0x0018, 0x01d9},
  		{0x0000, 0x1000, 0x01da},
  		{0x0500, 0x040a, 0x01db},
  		{0x0000, 0x0000, 0x01dc},
  		{0x0040, 0x0000, 0x0220},
  		{0x0000, 0x0000, 0x0228},
  		{0x0008, 0x0118, 0x0261},
  		{0x180a, 0x0108, 0x0271},
  		{0x0010, 0x0000, 0x02a8},
  		{0x0000, 0x0000, 0x02b0},
  		{0x0100, 0x0000, 0x02b8},
  		{0x0000, 0x0000, 0x02c0},
  		{0x0000, 0xa005, 0x02e8},
  		{0x7a40, 0x0146, 0x02e9},
  		{0x0000, 0x0100, 0x02ea},
  		{0x0000, 0x0000, 0x02eb},
  		{0x2800, 0xa000, 0x02f8},
  		{0xf640, 0x0147, 0x02f9},
  		{0x0000, 0x0100, 0x02fa},
  		{0x0000, 0x0000, 0x02fb},
  		{0x1000, 0xa000, 0x0300},
  		{0x0040, 0x0140, 0x0301},
  		{0x0000, 0x7100, 0x0302},
  		{0x0004, 0x0000, 0x0303},
  		{0x0000, 0xa000, 0x0308},
  		{0x0040, 0x0140, 0x0309},
  		{0x0000, 0x0100, 0x030a},
  		{0x0000, 0x0000, 0x030b},
  	};
  
  load_cfg((void*)cin, 0x20000, 168, 0, 0);
  config(0x0, 28, 0, 0);
  execute(0x7c04, 0, 0);
  }
  {
  /// ADORA.BlockStore %7, %arg1 [0] : memref<40xf32> -> memref<?xf32>  {Id = "3", KernelName = "kernel_mvt_1"}
  uint64_t dramoffset_3 = 0;
  uint64_t spadoffset_3 = 0;
  uint64_t roffset_3 = 0;
  store(arg_1 + dramoffset_3 + roffset_3, 0x1c000 + spadoffset_3, 160, 0, 0);
  spadoffset_3 = spadoffset_3 + 160;
  
  }
}
