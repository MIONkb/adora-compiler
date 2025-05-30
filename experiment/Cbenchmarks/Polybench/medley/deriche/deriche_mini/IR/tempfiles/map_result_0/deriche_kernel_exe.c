
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
/// kernel_deriche_5
volatile unsigned short cin_kernel_deriche_5[16][3] __attribute__((aligned(8))) = {
		{0x000a, 0x0118, 0x0251},
		{0x0110, 0x0000, 0x0298},
		{0x0000, 0x0000, 0x02a0},
		{0x2000, 0x0000, 0x02d8},
		{0x0041, 0x0100, 0x02d9},
		{0x0000, 0x0000, 0x02da},
		{0x0000, 0x0002, 0x02db},
		{0x3000, 0x0000, 0x02e0},
		{0x0041, 0x0100, 0x02e1},
		{0x0000, 0x0000, 0x02e2},
		{0x0000, 0x0882, 0x02e3},
		{0x0000, 0x0000, 0x02e4},
		{0x2800, 0x0000, 0x02e8},
		{0x0041, 0x0100, 0x02e9},
		{0x0000, 0x0000, 0x02ea},
		{0x0000, 0x0002, 0x02eb},
	};


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
    /// %1 = ADORA.BlockLoad %arg2 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_5"}
    uint64_t dramoffset_0 = 256 * int_14;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_2 + dramoffset_0 + roffset_0, 0x10000 + spadoffset_0, 8192, 0, _task_id, LD_DEP_EX_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 8192;
    
    }
    {
    /// %2 = ADORA.BlockLoad %arg3 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "1", KernelName = "kernel_deriche_5"}
    uint64_t dramoffset_1 = 256 * int_14;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_3 + dramoffset_1 + roffset_1, 0x12000 + spadoffset_1, 8192, 0, _task_id, LD_DEP_EX_LAST_TASK);
    spadoffset_1 = spadoffset_1 + 8192;
    
    }
    {
    /// kernel_deriche_5
    load_cfg((void*)cin_kernel_deriche_5, 0x20000, 96, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 16, _task_id, 0);
    execute(0x700, _task_id, EX_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %3, %arg1 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "2", KernelName = "kernel_deriche_5"}
    uint64_t dramoffset_2 = 256 * int_14;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_1 + dramoffset_2 + roffset_2, 0x14000 + spadoffset_2, 8192, _task_id, 0);
    spadoffset_2 = spadoffset_2 + 8192;
    
    }
    _task_id++;
  }
  
  
  
}
