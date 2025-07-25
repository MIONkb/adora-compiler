
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
/// merge_MATMUL
volatile unsigned short cin_merge_MATMUL[106][3] __attribute__((aligned(8))) = {
		{0x4800, 0x1000, 0x0018},
		{0xfe80, 0x0027, 0x0019},
		{0x0000, 0x0000, 0x001a},
		{0x0000, 0x0002, 0x001b},
		{0x0000, 0x1000, 0x0020},
		{0x0080, 0x0020, 0x0021},
		{0x0000, 0x0000, 0x0022},
		{0x0000, 0x0002, 0x0023},

    	/// output latency : [122, 117]
		{0x3800, 0x1000, 0x0028},
		{0x0040, 0x0020, 0x0029},
		{0x0000, 0x0000, 0x002a},
		// {0x0000, 0x0982/*[128 ,112]*/, 0x002b},
		{0x0000, 0x09e2/*[128 ,112]*/, 0x002b},
		{0x0000, 0x0000, 0x002c},

		{0x4800, 0x1000, 0x0030},
		{0xfe80, 0x0027, 0x0031},
		{0x0000, 0x0000, 0x0032},
		{0x0000, 0x0002, 0x0033},
		{0x5000, 0x1000, 0x0038},
		{0xfe80, 0x0027, 0x0039},
		{0x0000, 0x0000, 0x003a},
		{0x0000, 0x0002, 0x003b},
		{0x0000, 0x1000, 0x0040},
		{0x0080, 0x0020, 0x0041},
		{0x0000, 0x0000, 0x0042},
		{0x0000, 0x0002, 0x0043},
		{0x0000, 0x0000, 0x0068},
		{0x0030, 0x0000, 0x0070},
		{0x0000, 0x0010, 0x0078},
		{0x0000, 0x0010, 0x0080},
		{0x0000, 0x0000, 0x0090},
		{0x0021, 0x0400, 0x00b9},
		{0x0000, 0x0000, 0x00ba},
		{0x0010, 0x0105, 0x00bb},
		{0x0010, 0x0000, 0x00bc},
		{0x2101, 0x2300, 0x00c9},
		{0x0010, 0x0000, 0x00f1},
		{0x0000, 0x0010, 0x00f8},
		{0x0060, 0x0000, 0x00f9},
		{0x0000, 0x80c2, 0x0100},
		{0x0020, 0x0000, 0x0101},
		{0x0000, 0x3002, 0x0108},
		{0x0300, 0x0000, 0x0110},
		{0x0060, 0x0000, 0x0111},
		{0x0004, 0x0000, 0x0118},
		{0x0000, 0x0000, 0x0120},

    	/// merge latency : [117, 112]
		{0x22d2, 0x1a33, 0x0141},
		{0xd503, 0xf724, 0x0142},
		// {0x6448, 0x618f/*[128, 112]*/, 0x0143},
		{0x6448, 0x618d/*[128, 112]*/, 0x0143},
		{0xdc97, 0x0023, 0x0144},

		{0x4001, 0x0c00, 0x0149},
		{0x0003, 0x0b00, 0x0151},
		{0x0000, 0x8000, 0x0180},
		{0x0001, 0x0000, 0x0181},
		{0x0400, 0x3000, 0x0188},
		{0x0060, 0x0000, 0x0189},
		{0x8000, 0x0000, 0x0190},
		{0x0301, 0x0000, 0x0198},
		{0x0000, 0x3000, 0x01a0},
		{0x0060, 0x0000, 0x01a1},
		{0x0000, 0x4000, 0x01a8},
		{0x5101, 0x1900, 0x01d9},
		{0x0021, 0x0300, 0x01e1},
		{0x0000, 0x0000, 0x01e2},
		{0x0010, 0x0106, 0x01e3},
		{0x0010, 0x0000, 0x01e4},
		{0x0021, 0x0300, 0x01e9},
		{0x0000, 0x0000, 0x01ea},
		{0x0010, 0x0104, 0x01eb},
		{0x0010, 0x0000, 0x01ec},
		{0x0000, 0x0000, 0x0218},
		{0x000c, 0x0000, 0x0219},
		{0x0400, 0x00c1, 0x0220},
		{0x0100, 0x0000, 0x0228},
		{0x0002, 0x0000, 0x0229},
		{0x8100, 0x0000, 0x0230},
		{0x0000, 0x0000, 0x0238},
		{0x0000, 0x0000, 0x0239},
		{0x5101, 0x1a00, 0x0261},
		{0x0203, 0x0c00, 0x0269},
		{0x0003, 0x1c00, 0x0271},
		{0x1003, 0x2100, 0x0279},
		{0x0021, 0x0300, 0x0281},
		{0x0000, 0x0000, 0x0282},
		{0x0010, 0x0105, 0x0283},
		{0x0010, 0x0000, 0x0284},
		{0x0000, 0x0010, 0x02a0},
		{0x0000, 0x0010, 0x02b0},
		{0x0001, 0x000c, 0x02b8},
		{0x0000, 0x0001, 0x02c0},
		{0x0000, 0x0000, 0x02c8},
		{0x0000, 0x1000, 0x02e0},
		{0x0080, 0x0020, 0x02e1},
		{0x0000, 0x0000, 0x02e2},
		{0x0000, 0x0002, 0x02e3},
		{0x4800, 0x1000, 0x02f0},
		{0xfe80, 0x0027, 0x02f1},
		{0x0000, 0x0000, 0x02f2},
		{0x0000, 0x0002, 0x02f3},
		{0x0000, 0x1000, 0x02f8},
		{0x0080, 0x0020, 0x02f9},
		{0x0000, 0x0000, 0x02fa},
		{0x0000, 0x0002, 0x02fb},
		{0x2800, 0x1000, 0x0308},
		{0x0040, 0x0020, 0x0309},
		{0x0000, 0x0000, 0x030a},
		{0x0000, 0x0002, 0x030b},
	};


void merge_MATMUL(void* arg_0 ,void* arg_1 ,void* arg_2 ,void* arg_3){
  {
  /// %0 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "0", KernelName = "merge_MATMUL"}
  uint64_t dramoffset_0 = 0;
  uint64_t spadoffset_0 = 0;
  for(int idx_0 = 0; idx_0 < 4; idx_0++){
    uint64_t roffset_0 =  16*idx_0 ;
    load_data(arg_2 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 8, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 8;
  } 
  }
  {
  /// %1 = ADORA.BlockLoad %arg2 [0, 1] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "1", KernelName = "merge_MATMUL"}
  uint64_t dramoffset_1 = 4 * 1;
  uint64_t spadoffset_1 = 0;
  for(int idx_0 = 0; idx_0 < 4; idx_0++){
    uint64_t roffset_1 =  16*idx_0 ;
    load_data(arg_2 + dramoffset_1 + roffset_1, 0x10000 + spadoffset_1, 8, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_1 = spadoffset_1 + 8;
  } 
  }
  {
  /// %2 = ADORA.BlockLoad %arg2 [0, 2] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "2", KernelName = "merge_MATMUL"}
  uint64_t dramoffset_2 = 4 * 2;
  uint64_t spadoffset_2 = 0;
  for(int idx_0 = 0; idx_0 < 4; idx_0++){
    uint64_t roffset_2 =  16*idx_0 ;
    load_data(arg_2 + dramoffset_2 + roffset_2, 0x18000 + spadoffset_2, 8, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_2 = spadoffset_2 + 8;
  } 
  }
  {
  /// %3 = ADORA.BlockLoad %arg2 [0, 3] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "3", KernelName = "merge_MATMUL"}
  uint64_t dramoffset_3 = 4 * 3;
  uint64_t spadoffset_3 = 0;
  for(int idx_0 = 0; idx_0 < 4; idx_0++){
    uint64_t roffset_3 =  16*idx_0 ;
    load_data(arg_2 + dramoffset_3 + roffset_3, 0x0 + spadoffset_3, 8, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_3 = spadoffset_3 + 8;
  } 
  }
  {
  /// %4 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x4xi32> -> memref<4x4xi32>  {Id = "4", KernelName = "merge_MATMUL"}
  uint64_t dramoffset_4 = 0;
  uint64_t spadoffset_4 = 0;
  uint64_t roffset_4 = 0;
  load_data(arg_0 + dramoffset_4 + roffset_4, 0x1a000 + spadoffset_4, 64, 0, _task_id, LD_DEP_ST_LAST_TASK);
  spadoffset_4 = spadoffset_4 + 64;
  
  }
  {
  /// %5 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "5", KernelName = "merge_MATMUL"}
  uint64_t dramoffset_5 = 0;
  uint64_t spadoffset_5 = 0;
  for(int idx_0 = 0; idx_0 < 4; idx_0++){
    uint64_t roffset_5 =  16*idx_0 ;
    load_data(arg_1 + dramoffset_5 + roffset_5, 0x12000 + spadoffset_5, 8, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_5 = spadoffset_5 + 8;
  } 
  }
  {
  /// %6 = ADORA.BlockLoad %arg1 [0, 1] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "6", KernelName = "merge_MATMUL"}
  uint64_t dramoffset_6 = 4 * 1;
  uint64_t spadoffset_6 = 0;
  for(int idx_0 = 0; idx_0 < 4; idx_0++){
    uint64_t roffset_6 =  16*idx_0 ;
    load_data(arg_1 + dramoffset_6 + roffset_6, 0x2000 + spadoffset_6, 8, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_6 = spadoffset_6 + 8;
  } 
  }
  {
  /// %7 = ADORA.BlockLoad %arg1 [0, 2] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "7", KernelName = "merge_MATMUL"}
  uint64_t dramoffset_7 = 4 * 2;
  uint64_t spadoffset_7 = 0;
  for(int idx_0 = 0; idx_0 < 4; idx_0++){
    uint64_t roffset_7 =  16*idx_0 ;
    load_data(arg_1 + dramoffset_7 + roffset_7, 0xa000 + spadoffset_7, 8, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_7 = spadoffset_7 + 8;
  } 
  }
  {
  /// %8 = ADORA.BlockLoad %arg1 [0, 3] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "8", KernelName = "merge_MATMUL"}
  uint64_t dramoffset_8 = 4 * 3;
  uint64_t spadoffset_8 = 0;
  for(int idx_0 = 0; idx_0 < 4; idx_0++){
    uint64_t roffset_8 =  16*idx_0 ;
    load_data(arg_1 + dramoffset_8 + roffset_8, 0xc000 + spadoffset_8, 8, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_8 = spadoffset_8 + 8;
  } 
  }
  {
  /// merge_MATMUL
  load_cfg((void*)cin_merge_MATMUL, 0x20000, 636, _task_id, LD_DEP_EX_LAST_TASK);
  config(0x0, 106, _task_id, 0);
  execute(0x5afc, _task_id, EX_DEP_ST_LAST_TASK);
  }
  {
  /// ADORA.BlockStore %9, %arg3 [0, 0] : memref<4x4xi32> -> memref<?x4xi32>  {Id = "9", KernelName = "merge_MATMUL"}
  uint64_t dramoffset_9 = 0;
  uint64_t spadoffset_9 = 0;
  uint64_t roffset_9 = 0;
  store(arg_3 + dramoffset_9 + roffset_9, 0xe000 + spadoffset_9, 64, _task_id, 0);
  spadoffset_9 = spadoffset_9 + 64;
  
  }
  _task_id++;
  fence(1);
}
