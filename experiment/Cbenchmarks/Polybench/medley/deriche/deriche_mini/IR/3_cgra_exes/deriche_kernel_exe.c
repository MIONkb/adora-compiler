
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
/// kernel_deriche_3
volatile unsigned short cin_kernel_deriche_3[72][3] __attribute__((aligned(8))) = {
		{0x0800, 0x0004, 0x0008},
		{0x0841, 0x0106, 0x0009},
		{0x0000, 0x0000, 0x000a},
		{0x0000, 0x090a, 0x000b},
		{0x0020, 0x0000, 0x000c},
		{0x0000, 0x0000, 0x0010},
		{0x0001, 0x0100, 0x0011},
		{0x0000, 0x0000, 0x0012},
		{0x0000, 0x090a, 0x0013},
		{0x0000, 0x0000, 0x0014},
		{0x0000, 0x0004, 0x0030},
		{0x0841, 0x0106, 0x0031},
		{0x0000, 0x0000, 0x0032},
		{0x0000, 0x000a, 0x0033},
		{0x0033, 0x0000, 0x0058},
		{0x1000, 0x0000, 0x0068},
		{0x1000, 0x0030, 0x0070},
		{0x0000, 0x0000, 0x0078},
		{0x000a, 0x2200, 0x00a9},
		{0x100a, 0x1a00, 0x00b1},
		{0x35c4, 0xbe41, 0x00b8},
		{0x0008, 0x0200, 0x00b9},
		{0x00a0, 0x0300, 0x00c1},
		{0x0000, 0x0000, 0x00c2},
		{0x0050, 0x1000, 0x00c3},
		{0x4080, 0x0000, 0x00c4},
		{0x0000, 0x0000, 0x00e8},
		{0x0000, 0x0000, 0x00e9},
		{0x0000, 0x0000, 0x00f0},
		{0x0400, 0x0000, 0x00f8},
		{0x000c, 0x0000, 0x0101},
		{0x0200, 0x0000, 0x0108},
		{0x0040, 0x0000, 0x0109},
		{0x400a, 0x2200, 0x0131},
		{0x44fd, 0x3f57, 0x0138},
		{0x0008, 0x2000, 0x0139},
		{0xb54c, 0x3de1, 0x0148},
		{0x0008, 0x0200, 0x0149},
		{0x0080, 0x0000, 0x0179},
		{0x0000, 0x1000, 0x0180},
		{0x0000, 0x8000, 0x0188},
		{0x0001, 0x0000, 0x0189},
		{0x0000, 0x0000, 0x0190},
		{0x0060, 0x0000, 0x0199},
		{0x4598, 0xbf1b, 0x01c0},
		{0x0008, 0x0400, 0x01c1},
		{0x00a0, 0x0100, 0x01c9},
		{0x0000, 0x0000, 0x01ca},
		{0x0050, 0x1003, 0x01cb},
		{0x4080, 0x0000, 0x01cc},
		{0x000c, 0x0000, 0x0209},
		{0x4000, 0x0000, 0x0210},
		{0x0000, 0x0000, 0x0219},
		{0x0060, 0x0000, 0x0229},
		{0x00a0, 0x0100, 0x0259},
		{0x0000, 0x0000, 0x025a},
		{0x0050, 0x1000, 0x025b},
		{0x4080, 0x0000, 0x025c},
		{0x0300, 0x0000, 0x02a8},
		{0x0000, 0x0030, 0x02b8},
		{0x0000, 0x0020, 0x02c0},
		{0x0200, 0x0000, 0x02c8},
		{0x0000, 0x0000, 0x02f0},
		{0x0001, 0x0100, 0x02f1},
		{0x0000, 0x0000, 0x02f2},
		{0x0000, 0x088a, 0x02f3},
		{0x0000, 0x0000, 0x02f4},
		{0x0000, 0x0000, 0x0310},
		{0x0001, 0x0100, 0x0311},
		{0x0000, 0x0000, 0x0312},
		{0x0000, 0x08ea, 0x0313},
		{0x0000, 0x0000, 0x0314},
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
    /// %1 = ADORA.BlockLoad %arg1 [0, %arg4] : memref<?x64xf32> -> memref<64x32xf32>  {Id = "0", KernelName = "kernel_deriche_3"}
    uint64_t dramoffset_0 = 4 * int_14;
    uint64_t spadoffset_0 = 0;
    for(int idx_0 = 0; idx_0 < 64; idx_0++){
      uint64_t roffset_0 =  256*idx_0 ;
      load_data(arg_1 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 128, 0, _task_id, LD_DEP_EX_LAST_TASK);
      spadoffset_0 = spadoffset_0 + 128;
    } 
    }
    {
    /// kernel_deriche_3
    load_cfg((void*)cin_kernel_deriche_3, 0x20000, 432, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 72, _task_id, 0);
    execute(0x8823, _task_id, LD_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %5, %alloca_13 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_3"}
    store(&float_12, 0x18000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %4, %alloca_12 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_3"}
    store(&float_11, 0x0, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %3, %alloca_11 [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_3"}
    store(&float_10, 0x10000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %2, %arg2 [%arg4, 0] : memref<64x32xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_3"}
    uint64_t dramoffset_1 = 128 * int_14;
    uint64_t spadoffset_1 = 0;
    for(int idx_0 = 0; idx_0 < 64; idx_0++){
      uint64_t roffset_1 =  256*idx_0 ;
      store(arg_2 + dramoffset_1 + roffset_1, 0x2000 + spadoffset_1, 128, _task_id, 0);
      spadoffset_1 = spadoffset_1 + 128;
    } 
    }
    _task_id++;
  }
  
  
  
}
