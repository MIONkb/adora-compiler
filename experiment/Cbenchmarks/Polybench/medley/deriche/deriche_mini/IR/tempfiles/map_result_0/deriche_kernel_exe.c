
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
    /// %1 = ADORA.BlockLoad %arg0 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_0"}
    uint64_t dramoffset_0 = 256 * int_14;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 8192, 0, _task_id, LD_DEP_EX_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 8192;
    
    }
    {
    /// kernel_deriche_0
    volatile unsigned short cin[66][3] __attribute__((aligned(8))) = {
    		{0x0000, 0x0000, 0x0008},
    		{0x0001, 0x0100, 0x0009},
    		{0x0000, 0x0000, 0x000a},
    		{0x0000, 0x08ca, 0x000b},
    		{0x0020, 0x0000, 0x000c},
    		{0x3000, 0x0000, 0x0028},
    		{0x0041, 0x0100, 0x0029},
    		{0x0000, 0x0000, 0x002a},
    		{0x0000, 0x090a, 0x002b},
    		{0x0020, 0x0000, 0x002c},
    		{0x0000, 0x0000, 0x0030},
    		{0x0001, 0x0100, 0x0031},
    		{0x0000, 0x0000, 0x0032},
    		{0x0000, 0x090a, 0x0033},
    		{0x0020, 0x0000, 0x0034},
    		{0x0800, 0x0000, 0x0040},
    		{0x0001, 0x0100, 0x0041},
    		{0x0000, 0x0000, 0x0042},
    		{0x0000, 0x088a, 0x0043},
    		{0x0000, 0x0000, 0x0044},
    		{0x0002, 0x0000, 0x0058},
    		{0x0000, 0x0002, 0x0060},
    		{0x0000, 0x0003, 0x0068},
    		{0x0000, 0x0030, 0x0078},
    		{0x0001, 0x0001, 0x0080},
    		{0x0010, 0x0000, 0x0088},
    		{0x018a, 0x0118, 0x00c1},
    		{0x0050, 0x0018, 0x00c9},
    		{0x0000, 0x5000, 0x00ca},
    		{0x0300, 0x8010, 0x00cb},
    		{0x0040, 0x0000, 0x00cc},
    		{0x0000, 0x1000, 0x00f8},
    		{0x00c0, 0x6000, 0x0108},
    		{0x1004, 0x0030, 0x0110},
    		{0x0000, 0x0080, 0x0118},
    		{0x000a, 0x0098, 0x0151},
    		{0x0050, 0x0008, 0x0159},
    		{0x0000, 0x5000, 0x015a},
    		{0x0000, 0x8010, 0x015b},
    		{0x0040, 0x0000, 0x015c},
    		{0x44fd, 0x3f57, 0x0160},
    		{0x0008, 0x0008, 0x0161},
    		{0x0000, 0x1000, 0x0188},
    		{0x0040, 0x0000, 0x0198},
    		{0x0000, 0x0000, 0x01a0},
    		{0x100a, 0x0118, 0x01d9},
    		{0x4598, 0xbf1b, 0x01e0},
    		{0x0008, 0x0010, 0x01e1},
    		{0x0000, 0x9000, 0x0218},
    		{0x1040, 0x0000, 0x0220},
    		{0x0000, 0x0000, 0x0228},
    		{0x35c4, 0xbe41, 0x0260},
    		{0x0008, 0x0020, 0x0261},
    		{0x00d0, 0x0008, 0x0269},
    		{0x0000, 0x5000, 0x026a},
    		{0x0000, 0x8010, 0x026b},
    		{0x0040, 0x0000, 0x026c},
    		{0xb54c, 0x3de1, 0x0270},
    		{0x0008, 0x0018, 0x0271},
    		{0x0000, 0x000c, 0x02a8},
    		{0x0000, 0x0001, 0x02b0},
    		{0x0000, 0x0000, 0x02b8},
    		{0x2000, 0x0000, 0x02f8},
    		{0x0041, 0x0100, 0x02f9},
    		{0x0000, 0x0000, 0x02fa},
    		{0x0000, 0x000a, 0x02fb},
    	};
    
    load_cfg((void*)cin, 0x20000, 396, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 66, _task_id, 0);
    execute(0x10b1, _task_id, 0);
    }
    {
    /// ADORA.BlockStore %5, %alloca_12 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_0"}
    store(&float_11, 0x8000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %4, %alloca_11 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_0"}
    store(&float_10, 0xa000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %3, %alloca_14 [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_0"}
    store(&float_13, 0x0, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %2, %arg2 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_0"}
    uint64_t dramoffset_1 = 256 * int_14;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    store(arg_2 + dramoffset_1 + roffset_1, 0xc000 + spadoffset_1, 8192, _task_id, 0);
    spadoffset_1 = spadoffset_1 + 8192;
    
    }
    _task_id++;
  }
  
  
  
}
