
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
volatile unsigned short cin_unroll_MATMUL[30][3] __attribute__((aligned(8))) = {
		{0x2000, 0x9000, 0x0010},
		{0xf740, 0x8127, 0x0011},
		{0x4000, 0x0002, 0x0012},
		{0x0000, 0x0002, 0x0013},
		{0x8000, 0x9004, 0x0028},
		{0xc540, 0x8126, 0x0029},
		{0x4d78, 0x0002, 0x002a},
		{0x0000, 0x0002, 0x002b},
		{0x0000, 0x0000, 0x0060},
		{0x0000, 0x0000, 0x0070},
		{0x0035, 0x1100, 0x00a9},
		{0x0000, 0x0000, 0x00aa},
		{0x0010, 0x0903, 0x00ab},
		{0x1440, 0x0000, 0x00ac},
		{0x0020, 0x0000, 0x00f1},
		{0x8000, 0x0000, 0x0180},
		{0x2001, 0x1900, 0x01c9},
		{0x0400, 0x0000, 0x0210},
		{0x0000, 0x0000, 0x0219},
		{0x0000, 0x0004, 0x02a0},
		{0x0300, 0x0000, 0x02a8},
		{0x0000, 0x9000, 0x02e0},
		{0x0040, 0x8120, 0x02e1},
		{0x4000, 0x0002, 0x02e2},
		{0x0000, 0x0002, 0x02e3},
		{0x0800, 0x9000, 0x02f0},
		{0x0040, 0x8120, 0x02f1},
		{0x4000, 0x0002, 0x02f2},
		{0x0000, 0x08c2, 0x02f3},
		{0x0000, 0x0000, 0x02f4},
	};


void unroll_MATMUL(void* arg_0 ,void* arg_1 ,void* arg_2 ,void* arg_3){
  {
  /// %0 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x36xi32> -> memref<36x36xi32>  {Id = "0", KernelName = "unroll_MATMUL"}
  uint64_t dramoffset_0 = 0;
  uint64_t spadoffset_0 = 0;
  uint64_t roffset_0 = 0;
  load_data(arg_2 + dramoffset_0 + roffset_0, 0x10000 + spadoffset_0, 5184, 0, _task_id, LD_DEP_ST_LAST_TASK);
  spadoffset_0 = spadoffset_0 + 5184;
  
  }
  {
  /// %1 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x36xi32> -> memref<36x36xi32>  {Id = "1", KernelName = "unroll_MATMUL"}
  uint64_t dramoffset_1 = 0;
  uint64_t spadoffset_1 = 0;
  uint64_t roffset_1 = 0;
  load_data(arg_0 + dramoffset_1 + roffset_1, 0x0 + spadoffset_1, 5184, 0, _task_id, LD_DEP_ST_LAST_TASK);
  spadoffset_1 = spadoffset_1 + 5184;
  
  }
  {
  /// %2 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x36xi32> -> memref<36x36xi32>  {Id = "2", KernelName = "unroll_MATMUL"}
  uint64_t dramoffset_2 = 0;
  uint64_t spadoffset_2 = 0;
  uint64_t roffset_2 = 0;
  load_data(arg_1 + dramoffset_2 + roffset_2, 0x8000 + spadoffset_2, 5184, 0, _task_id, LD_DEP_ST_LAST_TASK);
  spadoffset_2 = spadoffset_2 + 5184;
  
  }
  {
  /// unroll_MATMUL
  load_cfg((void*)cin_unroll_MATMUL, 0x20000, 180, _task_id, LD_DEP_EX_LAST_TASK);
  config(0x0, 30, _task_id, 0);
  execute(0xa12, _task_id, EX_DEP_ST_LAST_TASK);
  }
  {
  /// ADORA.BlockStore %3, %arg3 [0, 0] : memref<36x36xi32> -> memref<?x36xi32>  {Id = "3", KernelName = "unroll_MATMUL"}
  uint64_t dramoffset_3 = 0;
  uint64_t spadoffset_3 = 0;
  uint64_t roffset_3 = 0;
  store(arg_3 + dramoffset_3 + roffset_3, 0x12000 + spadoffset_3, 5184, _task_id, 0);
  spadoffset_3 = spadoffset_3 + 5184;
  
  }
  _task_id++;
  fence(1);
}
void merge_MATMUL_4x4(void* arg_0 ,void* arg_1 ,void* arg_2){
  {
  /// %0 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x36xi32> -> memref<33x36xi32>  {Id = "0", KernelName = "merge_MATMUL_4x4"}
  uint64_t dramoffset_0 = 0;
  uint64_t spadoffset_0 = 0;
  uint64_t roffset_0 = 0;
  spadoffset_0 = spadoffset_0 + 4752;
  
  }
  {
  /// %1 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x36xi32> -> memref<36x33xi32>  {Id = "1", KernelName = "merge_MATMUL_4x4"}
  uint64_t dramoffset_1 = 0;
  uint64_t spadoffset_1 = 0;
  for(int idx_0 = 0; idx_0 < 36; idx