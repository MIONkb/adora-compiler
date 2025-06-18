
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void kernel_3mm(void* arg_0 ,void* arg_1 ,void* arg_2 ,void* arg_3 ,void* arg_4 ,void* arg_5 ,void* arg_6){
  {
  /// %0 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x20xf32> -> memref<16x20xf32>  {Id = "0", KernelName = "kernel_3mm_0"}
  uint64_t dramoffset_0 = 0;
  uint64_t spadoffset_0 = 0;
  uint64_t roffset_0 = 0;
  load_data(arg_1 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 1280, 0, /*task_id*/0, 0);
  spadoffset_0 = spadoffset_0 + 1280;
  
  }
  {
  /// %1 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x18xf32> -> memref<20x18xf32>  {Id = "1", KernelName = "kernel_3mm_0"}
  uint64_t dramoffset_1 = 0;
  uint64_t spadoffset_1 = 0;
  uint64_t roffset_1 = 0;
  load_data(arg_2 + dramoffset_1 + roffset_1, 0x1a000 + spadoffset_1, 1440, 0, /*task_id*/0, 0);
  spadoffset_1 = spadoffset_1 + 1440;
  
  }
  /// kernel_3mm_0
  {
  volatile unsigned short cin[22][3] __attribute__((aligned(8))) = {
  		{0x001a, 0x0020, 0x01e1},
  		{0x0000, 0x1000, 0x01e2},
  		{0x0500, 0x0405, 0x01e3},
  		{0x0000, 0x0000, 0x01e4},
  		{0x0001, 0x0000, 0x0229},
  		{0x0000, 0x0000, 0x0230},
  		{0x0008, 0x0118, 0x0279},
  		{0x2000, 0x0000, 0x02b8},
  		{0x0010, 0x0000, 0x02c0},
  		{0x0002, 0x0000, 0x02c8},
  		{0x1000, 0x5000, 0x02f8},
  		{0x0040, 0x8090, 0x02f9},
  		{0x0000, 0x6101, 0x02fa},
  		{0x1004, 0x0000, 0x02fb},
  		{0x4800, 0x5002, 0x0300},
  		{0xaac0, 0x8097, 0x0301},
  		{0x0f4c, 0x0101, 0x0302},
  		{0x0000, 0x0000, 0x0303},
  		{0x2000, 0x5000, 0x0308},
  		{0xfb40, 0x8097, 0x0309},
  		{0x0000, 0x0101, 0x030a},
  		{0x0000, 0x0000, 0x030b},
  	};
  
  load_cfg((void*)cin, 0x20000, 132, /*task_id*/0, 0);
  config(0x0, 22, /*task_id*/0, 0);
  execute(0x700b, /*task_id*/0, 0);
  }
  {
  /// ADORA.BlockStore %2, %arg0 [0, 0] : memref<16x18xf32> -> memref<?x18xf32>  {Id = "2", KernelName = "kernel_3mm_0"}
  uint64_t dramoffset_2 = 0;
  uint64_t spadoffset_2 = 0;
  uint64_t roffset_2 = 0;
  store(arg_0 + dramoffset_2 + roffset_2, 0x1c000 + spadoffset_2, 1152, /*task_id*/0, 0);
  spadoffset_2 = spadoffset_2 + 1152;
  
  }
  {
  /// %3 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x24xf32> -> memref<18x24xf32>  {Id = "0", KernelName = "kernel_3mm_1"}
  uint64_t dramoffset_0 = 0;
  uint64_t spadoffset_0 = 0;
  uint64_t roffset_0 = 0;
  load_data(arg_4 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 1728, 0, /*task_id*/1, 0);
  spadoffset_0 = spadoffset_0 + 1728;
  
  }
  {
  /// %4 = ADORA.BlockLoad %arg5 [0, 0] : memref<?x22xf32> -> memref<24x22xf32>  {Id = "1", KernelName = "kernel_3mm_1"}
  uint64_t dramoffset_1 = 0;
  uint64_t spadoffset_1 = 0;
  uint64_t roffset_1 = 0;
  load_data(arg_5 + dramoffset_1 + roffset_1, 0x2000 + spadoffset_1, 2112, 0, /*task_id*/1, 0);
  spadoffset_1 = spadoffset_1 + 2112;
  
  }
  /// kernel_3mm_1
  {
  volatile unsigned short cin[20][3] __attribute__((aligned(8))) = {
  		{0x2000, 0x6000, 0x0008},
  		{0xfa40, 0x80b7, 0x0009},
  		{0x2000, 0x0101, 0x000a},
  		{0x0000, 0x0000, 0x000b},
  		{0x1000, 0x6000, 0x0010},
  		{0x0040, 0x80b0, 0x0011},
  		{0x2000, 0x7101, 0x0012},
  		{0x1004, 0x0000, 0x0013},
  		{0xc800, 0x6002, 0x0020},
  		{0x81c0, 0x80b7, 0x0021},
  		{0x2ef8, 0x0101, 0x0022},
  		{0x0000, 0x0000, 0x0023},
  		{0x0000, 0x0000, 0x0058},
  		{0x1101, 0x0000, 0x0060},
  		{0x0000, 0x0000, 0x0068},
  		{0x001a, 0x0010, 0x00a1},
  		{0x0000, 0x1000, 0x00a2},
  		{0x0600, 0x0406, 0x00a3},
  		{0x0000, 0x0000, 0x00a4},
  		{0x0808, 0x0088, 0x00a9},
  	};
  
  load_cfg((void*)cin, 0x20000, 120, /*task_id*/1, 0);
  config(0x0, 20, /*task_id*/1, 0);
  execute(0x700b, /*task_id*/1, 0);
  }
  {
  /// ADORA.BlockStore %5, %arg3 [0, 0] : memref<18x22xf32> -> memref<?x22xf32>  {Id = "2", KernelName = "kernel_3mm_1"}
  uint64_t dramoffset_2 = 0;
  uint64_t spadoffset_2 = 0;
  uint64_t roffset_2 = 0;
  store(arg_3 + dramoffset_2 + roffset_2, 0x4000 + spadoffset_2, 1584, /*task_id*/1, 0);
  spadoffset_2 = spadoffset_2 + 1584;
  
  }
  {
  /// %6 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x18xf32> -> memref<16x18xf32>  {Id = "0", KernelName = "kernel_3mm_2"}
  uint64_t dramoffset_0 = 0;
  uint64_t spadoffset_0 = 0;
  uint64_t roffset_0 = 0;
  load_data(arg_0 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 1152, 0, /*task_id*/2, LD_DEP_ST_LAST_SEC_TASK);
  spadoffset_0 = spadoffset_0 + 1152;
  
  }
  {
  /// %7 = ADORA.BlockLoad %arg3 [0, 0] : memref<?x22xf32> -> memref<18x22xf32>  {Id = "1", KernelName = "kernel_3mm_2"}
  uint64_t dramoffset_1 = 0;
  uint64_t spadoffset_1 = 0;
  uint64_t roffset_1 = 0;
  load_data(arg_3 + dramoffset_1 + roffset_1, 0x2000 + spadoffset_1, 1584, 0, /*task_id*/2, LD_DEP_ST_LAST_TASK);
  spadoffset_1 = spadoffset_1 + 1584;
  
  }
  /// kernel_3mm_2
  volatile unsigned short cin[20][3] __attribute__((aligned(8))) = {
  		{0x2000, 0x4800, 0x0008},
  		{0xfbc0, 0x80b7, 0x0009},
  		{0x0000, 0x0101, 0x000a},
  		{0x0000, 0x0000, 0x000b},
  		{0x1000, 0x4800, 0x0010},
  		{0x0040, 0x80b0, 0x0011},
  		{0x0000, 0x7101, 0x0012},
  		{0x0004, 0x0000, 0x0013},
  		{0xc800, 0x4802, 0x0020},
  		{0xa2c0, 0x80b7, 0x0021},
  		{0x0f3a, 0x0101, 0x0022},
  		{0x0000, 0x0000, 0x0023},
  		{0x1010, 0x0000, 0x0058},
  		{0x8000, 0x0000, 0x0060},
  		{0x0000, 0x0000, 0x0068},
  		{0x001a, 0x0010, 0x0099},
  		{0x0000, 0x1000, 0x009a},
  		{0x8600, 0x0404, 0x009b},
  		{0x0000, 0x0000, 0x009c},
  		{0x0088, 0x0088, 0x00a1},
  	};
  
  load_cfg((void*)cin, 0x20000, 120, /*task_id*/2, 0);
  config(0x0, 20, /*task_id*/2, 0);
  execute(0x700b, /*task_id*/2, 0);
  {
  /// ADORA.BlockStore %8, %arg6 [0, 0] : memref<16x22xf32> -> memref<?x22xf32>  {Id = "2", KernelName = "kernel_3mm_2"}
  uint64_t dramoffset_2 = 0;
  uint64_t spadoffset_2 = 0;
  uint64_t roffset_2 = 0;
  store(arg_6 + dramoffset_2 + roffset_2, 0x4000 + spadoffset_2, 1408, /*task_id*/2, 0);
  spadoffset_2 = spadoffset_2 + 1408;
  
  }
}
