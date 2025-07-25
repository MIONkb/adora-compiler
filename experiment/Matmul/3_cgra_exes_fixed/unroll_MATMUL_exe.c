
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

static uint8_t _task_id = 0;

#define LD_DEP_ST_LAST_TASK 1     // this load command depends on the store command of last task
#define LD_DEP_EX_LAST_TASK 2     // this load command depends on the execute command of last task
#define LD_DEP_ST_LAST_SEC_TASK 3 // this load command depends on the store command of last second task
#define EX_DEP_ST_LAST_TASK 1     // this execute command depends on the store command of last task



//===----------------------------------------------------------------------===//
// Configuration Data 
//===----------------------------------------------------------------------===//
/// unroll_MATMUL
volatile unsigned short cin_unroll_MATMUL[27][3] __attribute__((aligned(8))) = {
		{0x0000, 0x1000, 0x0010},
		{0x0040, 0x8020, 0x0011},
		{0x4000, 0x0000, 0x0012},
		{0x0000, 0x0002, 0x0013},
		{0x1000, 0x1000, 0x0018},
		{0x0040, 0x8020, 0x0019},
		{0x4000, 0x0000, 0x001a},
		{0x0000, 0x08c2, 0x001b},
		{0x0020, 0x0000, 0x001c},
		{0x2800, 0x1000, 0x0020},
		{0xff40, 0x8027, 0x0021},
		{0x4000, 0x0000, 0x0022},
		{0x0000, 0x0002, 0x0023},
		{0x8000, 0x1000, 0x0028},
		{0xfd40, 0x8027, 0x0029},
		{0x4ff8, 0x0000, 0x002a},
		{0x0000, 0x0002, 0x002b},
		{0x0000, 0x0000, 0x0060},
		{0x0001, 0x0000, 0x0068},
		{0x0000, 0x0000, 0x0070},
		{0x0201, 0x2100, 0x00a9},
		{0x0003, 0x0a00, 0x00b1},
		{0x0000, 0x0000, 0x00f8},
		{0x0021, 0x0200, 0x0139},
		{0x0000, 0x0000, 0x013a},
		{0x0010, 0x0104, 0x013b},
		{0x0040, 0x0000, 0x013c},
	};


void unroll_MATMUL(void* arg_0 ,void* arg_1 ,void* arg_2 ,void* arg_3){
  {
  /// %0 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x4xi32> -> memref<4x4xi32>  {Id = "0", KernelName = "unroll_MATMUL"}
  uint64_t dramoffset_0 = 0;
  uint64_t spadoffset_0 = 0;
  uint64_t roffset_0 = 0;
  load_data(arg_2 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 64, 0, _task_id, LD_DEP_ST_LAST_TASK);
  spadoffset_0 = spadoffset_0 + 64;
  
  }
  {
  /// %1 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x4xi32> -> memref<4x4xi32>  {Id = "1", KernelName = "unroll_MATMUL"}
  uint64_t dramoffset_1 = 0;
  uint64_t spadoffset_1 = 0;
  uint64_t roffset_1 = 0;
  load_data(arg_0 + dramoffset_1 + roffset_1, 0x2000 + spadoffset_1, 64, 0, _task_id, LD_DEP_ST_LAST_TASK);
  spadoffset_1 = spadoffset_1 + 64;
  
  }
  {
  /// %2 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x4xi32> -> memref<4x4xi32>  {Id = "2", KernelName = "unroll_MATMUL"}
  uint64_t dramoffset_2 = 0;
  uint64_t spadoffset_2 = 0;
  uint64_t roffset_2 = 0;
  load_data(arg_1 + dramoffset_2 + roffset_2, 0x8000 + spadoffset_2, 64, 0, _task_id, LD_DEP_ST_LAST_TASK);
  spadoffset_2 = spadoffset_2 + 64;
  
  }
  {
  /// unroll_MATMUL
  load_cfg((void*)cin_unroll_MATMUL, 0x20000, 162, _task_id, LD_DEP_EX_LAST_TASK);
  config(0x0, 27, _task_id, 0);
  execute(0x1e, _task_id, EX_DEP_ST_LAST_TASK);
  }
  {
  /// ADORA.BlockStore %3, %arg3 [0, 0] : memref<4x4xi32> -> memref<?x4xi32>  {Id = "3", KernelName = "unroll_MATMUL"}
  uint64_t dramoffset_3 = 0;
  uint64_t spadoffset_3 = 0;
  uint64_t roffset_3 = 0;
  store(arg_3 + dramoffset_3 + roffset_3, 0x4000 + spadoffset_3, 64, _task_id, 0);
  spadoffset_3 = spadoffset_3 + 64;
  
  }
  _task_id++;
  fence(1);
}
