
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
/// kernel_deriche_2
volatile unsigned short cin_kernel_deriche_2[18][3] __attribute__((aligned(8))) = {
		{0x2000, 0x0000, 0x0008},
		{0x0041, 0x0100, 0x0009},
		{0x0000, 0x0000, 0x000a},
		{0x0000, 0x0002, 0x000b},
		{0x2800, 0x0000, 0x0018},
		{0x0041, 0x0100, 0x0019},
		{0x0000, 0x0000, 0x001a},
		{0x0000, 0x08a2, 0x001b},
		{0x0020, 0x0000, 0x001c},
		{0x2000, 0x0000, 0x0028},
		{0x0041, 0x0100, 0x0029},
		{0x0000, 0x0000, 0x002a},
		{0x0000, 0x0002, 0x002b},
		{0x0000, 0x0000, 0x0058},
		{0x0100, 0x0000, 0x0060},
		{0x0001, 0x0000, 0x0068},
		{0x0000, 0x0000, 0x0070},
		{0x100a, 0x1100, 0x00a9},
	};


/// kernel_deriche_5
volatile unsigned short cin_kernel_deriche_5[19][3] __attribute__((aligned(8))) = {
		{0x0010, 0x0000, 0x0221},
		{0x0000, 0x0001, 0x0228},
		{0x000a, 0x0c00, 0x0271},
		{0x0000, 0x0000, 0x02b0},
		{0x0001, 0x0000, 0x02c0},
		{0x0000, 0x0001, 0x02c8},
		{0x2800, 0x0000, 0x02f8},
		{0x0041, 0x0100, 0x02f9},
		{0x0000, 0x0000, 0x02fa},
		{0x0000, 0x0002, 0x02fb},
		{0x3000, 0x0000, 0x0308},
		{0x0041, 0x0100, 0x0309},
		{0x0000, 0x0000, 0x030a},
		{0x0000, 0x08a2, 0x030b},
		{0x0000, 0x0000, 0x030c},
		{0x2000, 0x0000, 0x0310},
		{0x0041, 0x0100, 0x0311},
		{0x0000, 0x0000, 0x0312},
		{0x0000, 0x0002, 0x0313},
	};


/// kernel_deriche_3
volatile unsigned short cin_kernel_deriche_3[64][3] __attribute__((aligned(8))) = {
		{0x0800, 0x0004, 0x0018},
		{0x0841, 0x0106, 0x0019},
		{0x0000, 0x0000, 0x001a},
		{0x0000, 0x090a, 0x001b},
		{0x0020, 0x0000, 0x001c},
		{0x0000, 0x0000, 0x0020},
		{0x0001, 0x0100, 0x0021},
		{0x0000, 0x0000, 0x0022},
		{0x0000, 0x090a, 0x0023},
		{0x0000, 0x0000, 0x0024},
		{0x0000, 0x0004, 0x0028},
		{0x0841, 0x0106, 0x0029},
		{0x0000, 0x0000, 0x002a},
		{0x0000, 0x000a, 0x002b},
		{0x0800, 0x0000, 0x0030},
		{0x0001, 0x0100, 0x0031},
		{0x0000, 0x0000, 0x0032},
		{0x0000, 0x088a, 0x0033},
		{0x0000, 0x0000, 0x0034},
		{0x0033, 0x0030, 0x0068},
		{0x0000, 0x0000, 0x0070},
		{0x0010, 0x0004, 0x0078},
		{0x0100, 0x0000, 0x0080},
		{0x35c4, 0xbe41, 0x00b0},
		{0x0008, 0x0200, 0x00b1},
		{0x00a0, 0x0300, 0x00b9},
		{0x0000, 0x0000, 0x00ba},
		{0x0050, 0x1003, 0x00bb},
		{0x4080, 0x0000, 0x00bc},
		{0x02a0, 0x0100, 0x00c1},
		{0x0000, 0x0000, 0x00c2},
		{0x0050, 0x1000, 0x00c3},
		{0x4080, 0x0000, 0x00c4},
		{0x00a0, 0x0100, 0x00c9},
		{0x0000, 0x0000, 0x00ca},
		{0x0050, 0x1000, 0x00cb},
		{0x4080, 0x0000, 0x00cc},
		{0x0000, 0x0000, 0x00f8},
		{0x0060, 0x0000, 0x00f9},
		{0x0100, 0x0000, 0x0100},
		{0x0000, 0x0100, 0x0108},
		{0x0020, 0x0000, 0x0109},
		{0x0000, 0x4000, 0x0110},
		{0x300a, 0x2400, 0x0141},
		{0x100a, 0x1100, 0x0149},
		{0x44fd, 0x3f57, 0x0150},
		{0x0008, 0x0100, 0x0151},
		{0x4598, 0xbf1b, 0x0158},
		{0x0008, 0x0800, 0x0159},
		{0x0060, 0x0000, 0x0189},
		{0x0004, 0x0000, 0x0190},
		{0x8000, 0x3000, 0x0198},
		{0x0000, 0x8000, 0x01a0},
		{0x0000, 0x0000, 0x01a1},
		{0x000a, 0x0a00, 0x01d9},
		{0xb54c, 0x3de1, 0x01e0},
		{0x0008, 0x0100, 0x01e1},
		{0x0060, 0x0000, 0x0219},
		{0x2000, 0x0000, 0x02a8},
		{0x0000, 0x0000, 0x02e8},
		{0x0001, 0x0100, 0x02e9},
		{0x0000, 0x0000, 0x02ea},
		{0x0000, 0x08aa, 0x02eb},
		{0x0020, 0x0000, 0x02ec},
	};


/// kernel_deriche_4
volatile unsigned short cin_kernel_deriche_4[83][3] __attribute__((aligned(8))) = {
		{0x0000, 0x0000, 0x0018},
		{0x0001, 0x0100, 0x0019},
		{0x0000, 0x0000, 0x001a},
		{0x0000, 0x08ca, 0x001b},
		{0x0020, 0x0000, 0x001c},
		{0x0000, 0x0000, 0x0028},
		{0x0001, 0x0100, 0x0029},
		{0x0000, 0x0000, 0x002a},
		{0x0000, 0x082a, 0x002b},
		{0x0020, 0x0000, 0x002c},
		{0x0002, 0x0000, 0x0068},
		{0x0000, 0x0002, 0x0070},
		{0x0000, 0x0002, 0x0078},
		{0x0000, 0x0003, 0x0080},
		{0x000a, 0x1c00, 0x00b9},
		{0x01a0, 0x0400, 0x00c1},
		{0x0000, 0x0000, 0x00c2},
		{0x0050, 0x1000, 0x00c3},
		{0x4080, 0x0000, 0x00c4},
		{0x0400, 0x0000, 0x0100},
		{0x0000, 0x0000, 0x0108},
		{0x0000, 0x0000, 0x0109},
		{0x4002, 0x0000, 0x0110},
		{0x0002, 0x0000, 0x0111},
		{0x0300, 0x0200, 0x0149},
		{0x1714, 0xbe3c, 0x0150},
		{0x0008, 0x0800, 0x0151},
		{0x010a, 0x2100, 0x0159},
		{0x0000, 0x0000, 0x0190},
		{0x0000, 0x0000, 0x0191},
		{0x0000, 0x0040, 0x0198},
		{0x0000, 0x0000, 0x01a0},
		{0x0022, 0x0000, 0x01a1},
		{0x0008, 0x0000, 0x01a8},
		{0x6028, 0x3dea, 0x01d8},
		{0x0008, 0x0200, 0x01d9},
		{0x00a0, 0x0100, 0x01e1},
		{0x0000, 0x0000, 0x01e2},
		{0x0050, 0x1000, 0x01e3},
		{0x4080, 0x0000, 0x01e4},
		{0x00a0, 0x0300, 0x01f1},
		{0x0000, 0x0000, 0x01f2},
		{0x0050, 0x1002, 0x01f3},
		{0x4080, 0x0000, 0x01f4},
		{0x0000, 0x00c0, 0x0220},
		{0x0004, 0x0000, 0x0221},
		{0x0000, 0x1000, 0x0228},
		{0x0008, 0x0000, 0x0229},
		{0x8000, 0x00c1, 0x0230},
		{0x0002, 0x0000, 0x0231},
		{0x0100, 0x0000, 0x0238},
		{0x0020, 0x0000, 0x0239},
		{0x4598, 0xbf1b, 0x0260},
		{0x0008, 0x0200, 0x0261},
		{0x02a0, 0x0200, 0x0271},
		{0x0000, 0x0000, 0x0272},
		{0x0050, 0x1000, 0x0273},
		{0x4080, 0x0000, 0x0274},
		{0x200a, 0x0900, 0x0279},
		{0x44fd, 0x3f57, 0x0280},
		{0x0008, 0x0800, 0x0281},
		{0x3000, 0x0000, 0x02b8},
		{0x0000, 0x0000, 0x02c0},
		{0x0300, 0x0000, 0x02c8},
		{0x0800, 0x0000, 0x02f8},
		{0x0001, 0x0100, 0x02f9},
		{0x0000, 0x0000, 0x02fa},
		{0x0000, 0x08ea, 0x02fb},
		{0x0020, 0x0000, 0x02fc},
		{0x1fe0, 0x03fc, 0x0300},
		{0xf841, 0x0101, 0x0301},
		{0x0000, 0x0000, 0x0302},
		{0x0000, 0x08ea, 0x0303},
		{0x0020, 0x0000, 0x0304},
		{0x07e0, 0x03fc, 0x0308},
		{0xf841, 0x0101, 0x0309},
		{0x0000, 0x0000, 0x030a},
		{0x0000, 0x000a, 0x030b},
		{0x1000, 0x0000, 0x0310},
		{0x0001, 0x0100, 0x0311},
		{0x0000, 0x0000, 0x0312},
		{0x0000, 0x086a, 0x0313},
		{0x0000, 0x0000, 0x0314},
	};


/// kernel_deriche_1
volatile unsigned short cin_kernel_deriche_1[75][3] __attribute__((aligned(8))) = {
		{0x0000, 0x0000, 0x0020},
		{0x0001, 0x0100, 0x0021},
		{0x0000, 0x0000, 0x0022},
		{0x0000, 0x08ea, 0x0023},
		{0x0020, 0x0000, 0x0024},
		{0xe83f, 0x03ff, 0x0028},
		{0x1fc1, 0x0100, 0x0029},
		{0x0000, 0x0000, 0x002a},
		{0x0000, 0x08ea, 0x002b},
		{0x0020, 0x0000, 0x002c},
		{0x0000, 0x0000, 0x0030},
		{0x0001, 0x0100, 0x0031},
		{0x0000, 0x0000, 0x0032},
		{0x0000, 0x086a, 0x0033},
		{0x0000, 0x0000, 0x0034},
		{0x1000, 0x0000, 0x0038},
		{0x0001, 0x0100, 0x0039},
		{0x0000, 0x0000, 0x003a},
		{0x0000, 0x08ca, 0x003b},
		{0x0000, 0x0000, 0x003c},
		{0x0000, 0x000c, 0x0070},
		{0x2031, 0x0008, 0x0078},
		{0x0020, 0x0001, 0x0080},
		{0x030a, 0x2200, 0x00b9},
		{0x44fd, 0x3f57, 0x00c0},
		{0x0008, 0x2000, 0x00c1},
		{0x4598, 0xbf1b, 0x00c8},
		{0x0008, 0x2000, 0x00c9},
		{0x0000, 0x0000, 0x0100},
		{0x0002, 0x0000, 0x0101},
		{0x0000, 0x0000, 0x0108},
		{0x0000, 0x0000, 0x0109},
		{0x0000, 0x0000, 0x0110},
		{0x0000, 0x0000, 0x0118},
		{0x100a, 0x2200, 0x0149},
		{0x00a0, 0x0100, 0x0151},
		{0x0000, 0x0000, 0x0152},
		{0x0050, 0x1002, 0x0153},
		{0x4080, 0x0000, 0x0154},
		{0x02a0, 0x0300, 0x0159},
		{0x0000, 0x0000, 0x015a},
		{0x0050, 0x1000, 0x015b},
		{0x4080, 0x0000, 0x015c},
		{0x0000, 0x0000, 0x0190},
		{0x0082, 0x0000, 0x0191},
		{0x0000, 0x1100, 0x0198},
		{0x0000, 0x1000, 0x01a0},
		{0x000a, 0x1300, 0x01d9},
		{0x0100, 0x0300, 0x01e1},
		{0x6028, 0x3dea, 0x01e8},
		{0x0008, 0x0300, 0x01e9},
		{0x0100, 0x0040, 0x0220},
		{0x000e, 0x0000, 0x0221},
		{0x4100, 0x0000, 0x0228},
		{0x0100, 0x0000, 0x0230},
		{0x1714, 0xbe3c, 0x0260},
		{0x0008, 0x0200, 0x0261},
		{0x02a0, 0x0300, 0x0269},
		{0x0000, 0x0000, 0x026a},
		{0x0050, 0x1000, 0x026b},
		{0x4080, 0x0000, 0x026c},
		{0x00a0, 0x0100, 0x0271},
		{0x0000, 0x0000, 0x0272},
		{0x0050, 0x1000, 0x0273},
		{0x4080, 0x0000, 0x0274},
		{0x0110, 0x0004, 0x02b0},
		{0xe03f, 0x03ff, 0x02f0},
		{0x1fc1, 0x0100, 0x02f1},
		{0x0000, 0x0000, 0x02f2},
		{0x0000, 0x000a, 0x02f3},
		{0x0000, 0x0000, 0x02f8},
		{0x0001, 0x0100, 0x02f9},
		{0x0000, 0x0000, 0x02fa},
		{0x0000, 0x082a, 0x02fb},
		{0x0000, 0x0000, 0x02fc},
	};


/// kernel_deriche_0
volatile unsigned short cin_kernel_deriche_0[68][3] __attribute__((aligned(8))) = {
		{0x0000, 0x0000, 0x0010},
		{0x0001, 0x0100, 0x0011},
		{0x0000, 0x0000, 0x0012},
		{0x0000, 0x088a, 0x0013},
		{0x0020, 0x0000, 0x0014},
		{0x2000, 0x0000, 0x0028},
		{0x0041, 0x0100, 0x0029},
		{0x0000, 0x0000, 0x002a},
		{0x0000, 0x000a, 0x002b},
		{0x0800, 0x0000, 0x0038},
		{0x0001, 0x0100, 0x0039},
		{0x0000, 0x0000, 0x003a},
		{0x0000, 0x08aa, 0x003b},
		{0x0000, 0x0000, 0x003c},
		{0x0002, 0x0000, 0x0060},
		{0x0000, 0x0002, 0x0068},
		{0x0000, 0x000c, 0x0070},
		{0x0000, 0x0020, 0x0078},
		{0x0030, 0x0000, 0x0080},
		{0x02a0, 0x0200, 0x00b1},
		{0x0000, 0x0000, 0x00b2},
		{0x0050, 0x1000, 0x00b3},
		{0x4080, 0x0000, 0x00b4},
		{0x35c4, 0xbe41, 0x00b8},
		{0x0008, 0x0800, 0x00b9},
		{0x44fd, 0x3f57, 0x00c0},
		{0x0008, 0x0400, 0x00c1},
		{0x0000, 0x0000, 0x0100},
		{0x0000, 0x0000, 0x0101},
		{0x0000, 0x0002, 0x0108},
		{0x0020, 0x0000, 0x0109},
		{0x0000, 0x0000, 0x0110},
		{0x0000, 0x0000, 0x0111},
		{0x0004, 0x0000, 0x0119},
		{0x0020, 0x0000, 0x0120},
		{0xb54c, 0x3de1, 0x0148},
		{0x0008, 0x0100, 0x0149},
		{0x000a, 0x0900, 0x0151},
		{0x00a0, 0x0300, 0x0159},
		{0x0000, 0x0000, 0x015a},
		{0x0050, 0x1003, 0x015b},
		{0x4080, 0x0000, 0x015c},
		{0x0000, 0x0000, 0x0198},
		{0x0100, 0x0000, 0x01a0},
		{0x0000, 0x00c0, 0x01a8},
		{0x0000, 0x0000, 0x01b0},
		{0x000a, 0x1200, 0x01d9},
		{0x400a, 0x2300, 0x01e1},
		{0x00a0, 0x0200, 0x01e9},
		{0x0000, 0x0000, 0x01ea},
		{0x0050, 0x1000, 0x01eb},
		{0x4080, 0x0000, 0x01ec},
		{0x0000, 0x0000, 0x0228},
		{0x0020, 0x0000, 0x0229},
		{0x0000, 0x0000, 0x0230},
		{0x4598, 0xbf1b, 0x0270},
		{0x0008, 0x0200, 0x0271},
		{0x2300, 0x0000, 0x02b8},
		{0x0000, 0x0000, 0x02f8},
		{0x0001, 0x0100, 0x02f9},
		{0x0000, 0x0000, 0x02fa},
		{0x0000, 0x090a, 0x02fb},
		{0x0020, 0x0000, 0x02fc},
		{0x2800, 0x0000, 0x0300},
		{0x0041, 0x0100, 0x0301},
		{0x0000, 0x0000, 0x0302},
		{0x0000, 0x090a, 0x0303},
		{0x0000, 0x0000, 0x0304},
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
    /// %1 = ADORA.BlockLoad %arg0 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_0"}
    uint64_t dramoffset_0 = 256 * int_14;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 8192, 0, _task_id, LD_DEP_EX_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 8192;
    
    }
    {
    /// kernel_deriche_0
    load_cfg((void*)cin_kernel_deriche_0, 0x20000, 408, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 68, _task_id, 0);
    execute(0x3052, _task_id, LD_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %5, %alloca_12 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_0"}
    store(&float_11, 0x18000, 8, _task_id, 0);

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
    store(arg_2 + dramoffset_1 + roffset_1, 0x1a000 + spadoffset_1, 8192, _task_id, 0);
    spadoffset_1 = spadoffset_1 + 8192;
    
    }
    _task_id++;
  }
  
  
  
  for (int int_15 = 0; int_15 < 64; int_15 = int_15 + 32){
    {
    /// %1 = ADORA.BlockLoad %arg0 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_1"}
    uint64_t dramoffset_0 = 256 * int_15;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x10000 + spadoffset_0, 8192, 0, _task_id, LD_DEP_EX_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 8192;
    
    }
    {
    /// kernel_deriche_1
    load_cfg((void*)cin_kernel_deriche_1, 0x20000, 450, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 75, _task_id, 0);
    execute(0x1878, _task_id, LD_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %6, %alloca_6 [] : memref<2xf32> -> memref<f32>  {Id = "5", KernelName = "kernel_deriche_1"}
    store(&float_5, 0x0, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %5, %alloca [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_1"}
    store(&float_4, 0x8000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %4, %alloca_9 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_1"}
    store(&float_8, 0x18000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %3, %arg3 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "2", KernelName = "kernel_deriche_1"}
    uint64_t dramoffset_2 = 256 * int_15;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_3 + dramoffset_2 + roffset_2, 0xa000 + spadoffset_2, 8192, _task_id, 0);
    spadoffset_2 = spadoffset_2 + 8192;
    
    }
    {
    /// ADORA.BlockStore %2, %alloca_10 [] : memref<2xf32> -> memref<f32>  {Id = "1", KernelName = "kernel_deriche_1"}
    store(&float_9, 0xc000, 8, _task_id, 0);

    }
    _task_id++;
  }
  
  
  
  for (int int_16 = 0; int_16 < 64; int_16 = int_16 + 32){
    {
    /// %1 = ADORA.BlockLoad %arg2 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_2"}
    uint64_t dramoffset_0 = 256 * int_16;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_2 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 8192, 0, _task_id, LD_DEP_EX_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 8192;
    
    }
    {
    /// %2 = ADORA.BlockLoad %arg3 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "1", KernelName = "kernel_deriche_2"}
    uint64_t dramoffset_1 = 256 * int_16;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_3 + dramoffset_1 + roffset_1, 0x8000 + spadoffset_1, 8192, 0, _task_id, LD_DEP_EX_LAST_TASK);
    spadoffset_1 = spadoffset_1 + 8192;
    
    }
    {
    /// kernel_deriche_2
    load_cfg((void*)cin_kernel_deriche_2, 0x20000, 108, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 18, _task_id, 0);
    execute(0x15, _task_id, LD_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %3, %arg1 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "2", KernelName = "kernel_deriche_2"}
    uint64_t dramoffset_2 = 256 * int_16;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_1 + dramoffset_2 + roffset_2, 0x2000 + spadoffset_2, 8192, _task_id, 0);
    spadoffset_2 = spadoffset_2 + 8192;
    
    }
    _task_id++;
  }
  
  
  
  for (int int_17 = 0; int_17 < 64; int_17 = int_17 + 32){
    {
    /// %1 = ADORA.BlockLoad %arg1 [0, %arg4] : memref<?x64xf32> -> memref<64x32xf32>  {Id = "0", KernelName = "kernel_deriche_3"}
    uint64_t dramoffset_0 = 4 * int_17;
    uint64_t spadoffset_0 = 0;
    for(int idx_0 = 0; idx_0 < 64; idx_0++){
      uint64_t roffset_0 =  256*idx_0 ;
      load_data(arg_1 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 128, 0, _task_id, LD_DEP_EX_LAST_TASK);
      spadoffset_0 = spadoffset_0 + 128;
    } 
    }
    {
    /// kernel_deriche_3
    load_cfg((void*)cin_kernel_deriche_3, 0x20000, 384, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 64, _task_id, 0);
    execute(0x43c, _task_id, LD_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %5, %alloca_13 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_3"}
    store(&float_12, 0x10000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %4, %alloca_12 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_3"}
    store(&float_11, 0x0, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %3, %alloca_11 [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_3"}
    store(&float_10, 0xa000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %2, %arg2 [0, %arg4] : memref<64x32xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_3"}
    uint64_t dramoffset_1 = 4 * int_17;
    uint64_t spadoffset_1 = 0;
    for(int idx_0 = 0; idx_0 < 64; idx_0++){
      uint64_t roffset_1 =  256*idx_0 ;
      store(arg_2 + dramoffset_1 + roffset_1, 0x2000 + spadoffset_1, 128, _task_id, 0);
      spadoffset_1 = spadoffset_1 + 128;
    } 
    }
    _task_id++;
  }
  
  
  
  for (int int_18 = 0; int_18 < 64; int_18 = int_18 + 32){
    {
    /// %1 = ADORA.BlockLoad %arg1 [0, %arg4] : memref<?x64xf32> -> memref<64x32xf32>  {Id = "0", KernelName = "kernel_deriche_4"}
    uint64_t dramoffset_0 = 4 * int_18;
    uint64_t spadoffset_0 = 0;
    for(int idx_0 = 0; idx_0 < 64; idx_0++){
      uint64_t roffset_0 =  256*idx_0 ;
      load_data(arg_1 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 128, 0, _task_id, LD_DEP_EX_LAST_TASK);
      spadoffset_0 = spadoffset_0 + 128;
    } 
    }
    {
    /// kernel_deriche_4
    load_cfg((void*)cin_kernel_deriche_4, 0x20000, 498, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 83, _task_id, 0);
    execute(0xf014, _task_id, LD_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %6, %alloca_8 [] : memref<2xf32> -> memref<f32>  {Id = "5", KernelName = "kernel_deriche_4"}
    store(&float_7, 0x0, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %5, %alloca_7 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_4"}
    store(&float_6, 0x8000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %4, %alloca_6 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_4"}
    store(&float_5, 0x1a000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %3, %alloca [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_4"}
    store(&float_4, 0x1c000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %2, %arg3 [0, %arg4] : memref<64x32xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_4"}
    uint64_t dramoffset_1 = 4 * int_18;
    uint64_t spadoffset_1 = 0;
    for(int idx_0 = 0; idx_0 < 64; idx_0++){
      uint64_t roffset_1 =  256*idx_0 ;
      store(arg_3 + dramoffset_1 + roffset_1, 0x1e000 + spadoffset_1, 128, _task_id, 0);
      spadoffset_1 = spadoffset_1 + 128;
    } 
    }
    _task_id++;
  }
  
  
  
  for (int int_19 = 0; int_19 < 64; int_19 = int_19 + 32){
    {
    /// %1 = ADORA.BlockLoad %arg2 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_5"}
    uint64_t dramoffset_0 = 256 * int_19;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_2 + dramoffset_0 + roffset_0, 0x18000 + spadoffset_0, 8192, 0, _task_id, LD_DEP_EX_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 8192;
    
    }
    {
    /// %2 = ADORA.BlockLoad %arg3 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "1", KernelName = "kernel_deriche_5"}
    uint64_t dramoffset_1 = 256 * int_19;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_3 + dramoffset_1 + roffset_1, 0x1a000 + spadoffset_1, 8192, 0, _task_id, LD_DEP_EX_LAST_TASK);
    spadoffset_1 = spadoffset_1 + 8192;
    
    }
    {
    /// kernel_deriche_5
    load_cfg((void*)cin_kernel_deriche_5, 0x20000, 114, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 19, _task_id, 0);
    execute(0xd000, _task_id, LD_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %3, %arg1 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "2", KernelName = "kernel_deriche_5"}
    uint64_t dramoffset_2 = 256 * int_19;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_1 + dramoffset_2 + roffset_2, 0x1c000 + spadoffset_2, 8192, _task_id, 0);
    spadoffset_2 = spadoffset_2 + 8192;
    
    }
    _task_id++;
  }
  
  
  
}
