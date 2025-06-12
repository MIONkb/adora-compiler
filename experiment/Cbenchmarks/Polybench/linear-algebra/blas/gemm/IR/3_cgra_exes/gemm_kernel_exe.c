
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

uint8_t _task_id = 0;

#define LD_DEP_ST_LAST_TASK 1     // this load command depends on the store command of last task
#define LD_DEP_EX_LAST_TASK 2     // this load command depends on the execute command of last task
#define LD_DEP_ST_LAST_SEC_TASK 3 // this load command depends on the store command of last second task
#define EX_DEP_ST_LAST_TASK 1     // this EXECUTE command depends on the store command of last task



//===----------------------------------------------------------------------===//
// Configuration Data 
//===----------------------------------------------------------------------===//
/// kernel_gemm_1
volatile unsigned short cin_kernel_gemm_1[33][3] __attribute__((aligned(8))) = {
		{0x2000, 0x7800, 0x0008},
		{0xf8c0, 0x00cf, 0x0009},
		{0x0000, 0x0000, 0x000a},
		{0x0000, 0x0002, 0x000b},
		{0x0800, 0x7800, 0x0020},
		{0x0040, 0x00c8, 0x0021},
		{0x0000, 0x0000, 0x0022},
		{0x0000, 0x0962, 0x0023},
		{0x0020, 0x0000, 0x0024},
		{0x2800, 0x7803, 0x0028},
		{0x4b00, 0x00cf, 0x0029},
		{0x0000, 0x0000, 0x002a},
		{0x0000, 0x0002, 0x002b},
		{0x0000, 0x7800, 0x0038},
		{0x0040, 0x00c8, 0x0039},
		{0x0000, 0x0000, 0x003a},
		{0x0000, 0x0002, 0x003b},
		{0x0000, 0x0000, 0x0058},
		{0x0000, 0x0004, 0x0060},
		{0x0100, 0x0000, 0x0068},
		{0x0000, 0x0000, 0x0070},
		{0x8000, 0x0000, 0x0078},
		{0x0000, 0x0000, 0x0080},
		{0x0000, 0x3fc0, 0x00a0},
		{0x000a, 0x0010, 0x00a1},
		{0x300a, 0x0110, 0x00b1},
		{0x700c, 0x0130, 0x00b9},
		{0x0000, 0x2000, 0x00f8},
		{0x0080, 0x0000, 0x0100},
		{0x002c, 0x0100, 0x0139},
		{0x0000, 0x2000, 0x013a},
		{0x1200, 0xc80f, 0x013b},
		{0x0000, 0x0000, 0x013c},
	};


/// kernel_gemm_0
volatile unsigned short cin_kernel_gemm_0[13][3] __attribute__((aligned(8))) = {
		{0x999a, 0x3f99, 0x0268},
		{0x000a, 0x0040, 0x0269},
		{0x0000, 0x0000, 0x02b0},
		{0x0000, 0x0000, 0x02b8},
		{0x2000, 0x6400, 0x02f0},
		{0x0000, 0x0000, 0x02f1},
		{0x0000, 0x0000, 0x02f2},
		{0x0000, 0x08c2, 0x02f3},
		{0x0020, 0x0000, 0x02f4},
		{0x2000, 0x6400, 0x0300},
		{0x0000, 0x0000, 0x0301},
		{0x0000, 0x0000, 0x0302},
		{0x0000, 0x0002, 0x0303},
	};


void gemm(void* arg_0 ,void* arg_1 ,void* arg_2){
  for (int int_3 = 0; int_3 < 20; int_3 = int_3 + 1){
    {
    /// %0 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
    uint64_t dramoffset_0 = 100 * int_3;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 100, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 100;
    
    }
    {
    /// kernel_gemm_0
    load_cfg((void*)cin_kernel_gemm_0, 0x20000, 78, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 13, _task_id, 0);
    execute(0x2800, _task_id, EX_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %1, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
    uint64_t dramoffset_1 = 100 * int_3;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    store(arg_0 + dramoffset_1 + roffset_1, 0x10000 + spadoffset_1, 100, _task_id, 0);
    spadoffset_1 = spadoffset_1 + 100;
    
    }
    _task_id++;
    {
    /// %2 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
    uint64_t dramoffset_0 = 100 * int_3;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 100, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 100;
    
    }
    {
    /// %3 = ADORA.BlockLoad %arg1 [%arg3, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
    uint64_t dramoffset_1 = 120 * int_3;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_1 + dramoffset_1 + roffset_1, 0x0 + spadoffset_1, 120, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_1 = spadoffset_1 + 120;
    
    }
    {
    /// %4 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
    uint64_t dramoffset_2 = 0;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    load_data(arg_2 + dramoffset_2 + roffset_2, 0xa000 + spadoffset_2, 3000, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_2 = spadoffset_2 + 3000;
    
    }
    {
    /// kernel_gemm_1
    load_cfg((void*)cin_kernel_gemm_1, 0x20000, 198, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 33, _task_id, 0);
    execute(0x59, _task_id, EX_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %5, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
    uint64_t dramoffset_3 = 100 * int_3;
    uint64_t spadoffset_3 = 0;
    uint64_t roffset_3 = 0;
    store(arg_0 + dramoffset_3 + roffset_3, 0x2000 + spadoffset_3, 100, _task_id, 0);
    spadoffset_3 = spadoffset_3 + 100;
    
    }
    _task_id++;
  }
  
  
  
}
