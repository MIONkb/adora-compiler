
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
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x10000 + spadoffset_0, 8192, 0, _task_id, LD_DEP_EX_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 8192;
    
    }
    {
    /// kernel_deriche_0
    volatile unsigned short cin[65][3] __attribute__((aligned(8))) = {
    		{0x0000, 0x0000, 0x0030},
    		{0x0001, 0x0100, 0x0031},
    		{0x0000, 0x0000, 0x0032},
    		{0x0000, 0x08aa, 0x0033},
    		{0x0000, 0x0000, 0x0034},
    		{0x0000, 0x000c, 0x0070},
    		{0x0120, 0x0000, 0x0078},
    		{0x0050, 0x0008, 0x00c1},
    		{0x0000, 0x5000, 0x00c2},
    		{0x0000, 0x8010, 0x00c3},
    		{0x0040, 0x0000, 0x00c4},
    		{0x0000, 0x0c10, 0x0100},
    		{0x0000, 0x0084, 0x0108},
    		{0x0001, 0x0000, 0x0109},
    		{0x000a, 0x0098, 0x0141},
    		{0x44fd, 0x3f57, 0x0148},
    		{0x0008, 0x0010, 0x0149},
    		{0x0050, 0x0018, 0x0151},
    		{0x0000, 0x5000, 0x0152},
    		{0x0300, 0x8010, 0x0153},
    		{0x0040, 0x0000, 0x0154},
    		{0x0040, 0x0000, 0x0188},
    		{0x0000, 0x0000, 0x0190},
    		{0x2040, 0x0010, 0x0198},
    		{0xb54c, 0x3de1, 0x01c0},
    		{0x0008, 0x0018, 0x01c1},
    		{0x010a, 0x0118, 0x01c9},
    		{0x200a, 0x0088, 0x01d9},
    		{0x4598, 0xbf1b, 0x01e0},
    		{0x0008, 0x0008, 0x01e1},
    		{0x0040, 0x0000, 0x0208},
    		{0x0000, 0x0000, 0x0210},
    		{0x0004, 0x0000, 0x0218},
    		{0x0000, 0x0000, 0x0229},
    		{0x0150, 0x0020, 0x0249},
    		{0x0000, 0x5000, 0x024a},
    		{0x0000, 0x8010, 0x024b},
    		{0x0040, 0x0000, 0x024c},
    		{0x35c4, 0xbe41, 0x0258},
    		{0x0008, 0x0018, 0x0259},
    		{0x0000, 0x0000, 0x0298},
    		{0x0010, 0x0010, 0x02a0},
    		{0x0000, 0x0020, 0x02a8},
    		{0x0000, 0x0020, 0x02b0},
    		{0x2300, 0x0020, 0x02b8},
    		{0x0200, 0x0000, 0x02c0},
    		{0x2000, 0x0000, 0x02e0},
    		{0x0041, 0x0100, 0x02e1},
    		{0x0000, 0x0000, 0x02e2},
    		{0x0000, 0x000a, 0x02e3},
    		{0x0000, 0x0000, 0x02f8},
    		{0x0001, 0x0100, 0x02f9},
    		{0x0000, 0x0000, 0x02fa},
    		{0x0000, 0x090a, 0x02fb},
    		{0x0020, 0x0000, 0x02fc},
    		{0x3000, 0x0000, 0x0300},
    		{0x0041, 0x0100, 0x0301},
    		{0x0000, 0x0000, 0x0302},
    		{0x0000, 0x090a, 0x0303},
    		{0x0000, 0x0000, 0x0304},
    		{0x0800, 0x0000, 0x0308},
    		{0x0001, 0x0100, 0x0309},
    		{0x0000, 0x0000, 0x030a},
    		{0x0000, 0x08aa, 0x030b},
    		{0x0000, 0x0000, 0x030c},
    	};
    
    load_cfg((void*)cin, 0x20000, 390, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 65, _task_id, 0);
    execute(0x7220, _task_id, EX_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %5, %alloca_12 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_0"}
    store(&float_11, 0x18000, 8, _task_id, 0);

    }
    {
  