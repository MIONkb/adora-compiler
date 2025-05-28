
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
    volatile unsigned short cin[64][3] __attribute__((aligned(8))) = {
    		{0x0000, 0x0000, 0x0038},
    		{0x0001, 0x0100, 0x0039},
    		{0x0000, 0x0000, 0x003a},
    		{0x0000, 0x08aa, 0x003b},
    		{0x0000, 0x0000, 0x003c},
    		{0x0030, 0x0000, 0x0080},
    		{0x0000, 0x8000, 0x00f8},
    		{0x1000, 0x0000, 0x0100},
    		{0x0000, 0x1000, 0x0110},
    		{0x0050, 0x0008, 0x0149},
    		{0x0000, 0x5000, 0x014a},
    		{0x0000, 0x8010, 0x014b},
    		{0x0040, 0x0000, 0x014c},
    		{0x4598, 0xbf1b, 0x0150},
    		{0x0008, 0x0018, 0x0151},
    		{0x0000, 0x0c00, 0x0188},
    		{0x0000, 0x0084, 0x0190},
    		{0x0000, 0x0000, 0x0198},
    		{0x0001, 0x0000, 0x0199},
    		{0x0000, 0x1010, 0x01a0},
    		{0x44fd, 0x3f57, 0x01d0},
    		{0x0008, 0x0010, 0x01d1},
    		{0x0050, 0x0020, 0x01d9},
    		{0x0000, 0x5000, 0x01da},
    		{0x0300, 0x8010, 0x01db},
    		{0x0040, 0x0000, 0x01dc},
    		{0x100a, 0x00a0, 0x01e1},
    		{0xb54c, 0x3de1, 0x01e8},
    		{0x0008, 0x0020, 0x01e9},
    		{0x0000, 0x0000, 0x0220},
    		{0xc000, 0x0000, 0x0228},
    		{0x0000, 0x0000, 0x0229},
    		{0x0000, 0x1000, 0x0230},
    		{0x0000, 0x0000, 0x0238},
    		{0x000a, 0x0050, 0x0269},
    		{0x180a, 0x0058, 0x0271},
    		{0x35c4, 0xbe41, 0x0278},
    		{0x0008, 0x0100, 0x0279},
    		{0x0150, 0x0018, 0x0281},
    		{0x0000, 0x5000, 0x0282},
    		{0x0000, 0x8010, 0x0283},
    		{0x0040, 0x0000, 0x0284},
    		{0x0300, 0x0000, 0x02b8},
    		{0x0000, 0x0000, 0x02c0},
    		{0x0212, 0x0000, 0x02c8},
    		{0x3800, 0x0000, 0x02f8},
    		{0x0041, 0x0100, 0x02f9},
    		{0x0000, 0x0000, 0x02fa},
    		{0x0000, 0x090a, 0x02fb},
    		{0x0020, 0x0000, 0x02fc},
    		{0x1000, 0x0000, 0x0300},
    		{0x0001, 0x0100, 0x0301},
    		{0x0000, 0x0000, 0x0302},
    		{0x0000, 0x088a, 0x0303},
    		{0x0000, 0x0000, 0x0304},
    		{0x2000, 0x0000, 0x0308},
    		{0x0041, 0x0100, 0x0309},
    		{0x0000, 0x0000, 0x030a},
    		{0x0000, 0x000a, 0x030b},
    		{0x0800, 0x0000, 0x0310},
    		{0x0001, 0x0100, 0x0311},
    		{0x0000, 0x0000, 0x0312},
    		{0x0000, 0x090a, 0x0313},
    		{0x0000, 0x0000, 0x0314},
    	};
    
    load_cfg((void*)cin, 0x20000, 384, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 64, _task_id, 0);
    execute(0xf040, _task_id, EX_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %5, %alloca_12 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_0"}
    store(&float_11, 0x1a000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %4, %alloca_11 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_0"}
    store(&float_10, 0x1c000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %3, %alloca_14 [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_0"}
    store(&float_13, 0x8000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %2, %arg2 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_0"}
    uint64_t dramoffset_1 = 256 * int_14;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    store(arg_2 + dramoffset_1 + roffset_1, 0x1e000 + spadoffset_1, 8192, _task_id, 0);
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
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 8192, 0, _task_id, LD_DEP_EX_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 8192;
    
    }
    {
    /// kernel_deriche_1
    volatile unsigned short cin[78][3] __attribute__((aligned(8))) = {
    		{0x1000, 0x0000, 0x0008},
    		{0x0001, 0x0100, 0x0009},
    		{0x0000, 0x0000, 0x000a},
    		{0x0000, 0x086a, 0x000b},
    		{0x0020, 0x0000, 0x000c},
    		{0x0800, 0x0000, 0x0010},
    		{0x0001, 0x0100, 0x0011},
    		{0x0000, 0x0000, 0x0012},
    		{0x0000, 0x08ea, 0x0013},
    		{0x0020, 0x0000, 0x0014},
    		// {0xe000, 0x03ff, 0x0018},
			{0xe03F, 0x03ff, 0x0018},
    		{0x1fc1, 0x0100, 0x0019},
    		{0x0000, 0x0000, 0x001a},
    		{0x0000, 0x000a, 0x001b},
    		// {0xf800, 0x03ff, 0x0020},
			{0xf83f, 0x03ff, 0x0020},
    		{0x1fc1, 0x0100, 0x0021},
    		{0x0000, 0x0000, 0x0022},
    		{0x0000, 0x08ea, 0x0023},
    		{0x0000, 0x0000, 0x0024},
    		{0x0000, 0x0000, 0x0028},
    		{0x0001, 0x0100, 0x0029},
    		{0x0000, 0x0000, 0x002a},
    		{0x0000, 0x084a, 0x002b},
    		{0x0000, 0x0000, 0x002c},
    		{0x0003, 0x0000, 0x0058},
    		{0x0000, 0x0010, 0x0060},
    		{0x0010, 0x0000, 0x0068},
    		{0x0030, 0x0000, 0x0070},
    		{0x0150, 0x0020, 0x0099},
    		{0x0000, 0x5000, 0x009a},
    		{0x0000, 0x8010, 0x009b},
    		{0x0040, 0x0000, 0x009c},
    		{0x44fd, 0x3f57, 0x00a0},
    		{0x0008, 0x0100, 0x00a1},
    		{0x018a, 0x0118, 0x00a9},
    		{0x0080, 0x0008, 0x00b1},
    		{0x0000, 0x0000, 0x00e0},
    		{0x0000, 0x2000, 0x00e8},
    		{0x0084, 0x0000, 0x00f0},
    		{0x0003, 0x0000, 0x00f1},
    		{0x0004, 0x0004, 0x00f8},
    		{0x0000, 0x000c, 0x0100},
    		{0x0000, 0x0080, 0x0108},
    		{0x4598, 0xbf1b, 0x0128},
    		{0x0008, 0x0040, 0x0129},
    		{0x0050, 0x0010, 0x0131},
    		{0x0000, 0x5000, 0x0132},
    		{0x0200, 0x8010, 0x0133},
    		{0x0040, 0x0000, 0x0134},
    		{0x008a, 0x0050, 0x0139},
    		{0x000a, 0x0110, 0x0141},
    		{0x0050, 0x0008, 0x0149},
    		{0x0000, 0x5000, 0x014a},
    		{0x0000, 0x8010, 0x014b},
    		{0x0040, 0x0000, 0x014c},
    		{0x6028, 0x3dea, 0x0150},
    		{0x0008, 0x0018, 0x0151},
    		{0x0003, 0x0000, 0x0181},
    		{0x0000, 0x8000, 0x0188},
    		{0x1004, 0x0000, 0x0190},
    		{0x0001, 0x0000, 0x0191},
    		{0x0040, 0x0000, 0x0198},
    		{0x1714, 0xbe3c, 0x01d0},
    		{0x0008, 0x0010, 0x01d1},
    		{0x0050, 0x0008, 0x01d9},
    		{0x0000, 0x5000, 0x01da},
    		{0x0000, 0x8010, 0x01db},
    		{0x0040, 0x0000, 0x01dc},
    		{0x0003, 0x0000, 0x0211},
    		{0x0000, 0x0c00, 0x0218},
    		{0x2000, 0x0080, 0x0220},
    		{0x0080, 0x0008, 0x0269},
    		{0x2000, 0x0000, 0x02a0},
    		{0x0000, 0x0000, 0x02e0},
    		{0x0001, 0x0100, 0x02e1},
    		{0x0000, 0x0000, 0x02e2},
    		{0x0000, 0x08aa, 0x02e3},
    		{0x0020, 0x0000, 0x02e4},
    	};

	printf("cin: %x\n", cin);
    
    load_cfg((void*)cin, 0x20000, 468, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 78, _task_id, 0);
    execute(0x21f, _task_id, EX_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %6, %alloca_6 [] : memref<2xf32> -> memref<f32>  {Id = "5", KernelName = "kernel_deriche_1"}
    store(&float_5, 0x2000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %5, %alloca [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_1"}
    store(&float_4, 0x4000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %4, %alloca_9 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_1"}
    store(&float_8, 0x8000, 8, _task_id, 0);

    }
    {
    /// ADORA.BlockStore %3, %arg3 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "2", KernelName = "kernel_deriche_1"}
    uint64_t dramoffset_2 = 256 * int_15;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_3 + dramoffset_2 + roffset_2, 0x6000 + spadoffset_2, 8192, _task_id, 0);
    spadoffset_2 = spadoffset_2 + 8192;
    
    }
    {
    /// ADORA.BlockStore %2, %alloca_10 [] : memref<2xf32> -> memref<f32>  {Id = "1", KernelName = "kernel_deriche_1"}
    store(&float_9, 0x10000, 8, _task_id, 0);

    }
    _task_id++;
  }
  
  
  
}
