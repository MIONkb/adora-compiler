
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"
#include "include/encoding.h"
uint8_t _task_id = 0;

#define LD_DEP_ST_LAST_TASK 1     // this load command depends on the store command of last task
#define LD_DEP_EX_LAST_TASK 2     // this load command depends on the execute command of last task
#define LD_DEP_ST_LAST_SEC_TASK 3 // this load command depends on the store command of last second task
#define EX_DEP_ST_LAST_TASK 1     // this EXECUTE command depends on the store command of last task



//===----------------------------------------------------------------------===//
// Configuration Data 
//===----------------------------------------------------------------------===//
/// kernel_gemm_1
volatile unsigned short cin_kernel_gemm_1[122][3] __attribute__((aligned(8))) = {
		{0xc803, 0x1400, 0x0008},
		{0xfa00, 0x00cf, 0x0009},
		{0x0000, 0x0000, 0x000a},
		{0x0000, 0x0002, 0x000b},
		{0x1800, 0x1400, 0x0010},
		{0x0040, 0x00c8, 0x0011},
		{0x0000, 0x0000, 0x0012},
		{0x0000, 0x0a42, 0x0013},
		{0x0020, 0x0000, 0x0014},
		{0xd064, 0x1412, 0x0018},
		{0x6a40, 0x00cf, 0x0019},
		{0x0000, 0x0000, 0x001a},
		{0x0000, 0x0002, 0x001b},
		{0x0000, 0x1400, 0x0020},
		{0x0040, 0x00c8, 0x0021},
		{0x0000, 0x0000, 0x0022},
		{0x0000, 0x0062, 0x0023},
		{0xc002, 0x1400, 0x0028},
		{0xfa00, 0x00cf, 0x0029},
		{0x0000, 0x0000, 0x002a},
		{0x0000, 0x0002, 0x002b},
		{0xd032, 0x1412, 0x0030},
		{0x6a40, 0x00cf, 0x0031},
		{0x0000, 0x0000, 0x0032},
		{0x0000, 0x0002, 0x0033},
		{0xc800, 0x1412, 0x0038},
		{0x6a40, 0x00cf, 0x0039},
		{0x0000, 0x0000, 0x003a},
		{0x0000, 0x0002, 0x003b},
		{0x0000, 0x0000, 0x0058},
		{0x0000, 0x0010, 0x0060},
		{0x0000, 0x0000, 0x0070},
		{0x1000, 0x0030, 0x0078},
		{0x0000, 0x0000, 0x0080},
		{0x0000, 0x3fc0, 0x00a0},
		{0x000a, 0x0100, 0x00a1},
		{0xa00c, 0x2300, 0x00a9},
		{0x300a, 0x1b00, 0x00b1},
		{0x200a, 0x2200, 0x00b9},
		{0x0000, 0x3fc0, 0x00c0},
		{0x000a, 0x0100, 0x00c1},
		{0x0300, 0x0100, 0x00f0},
		{0x000c, 0x0000, 0x00f1},
		{0x0304, 0x0000, 0x00f8},
		{0x0000, 0x2000, 0x0100},
		{0x0004, 0x0000, 0x0108},
		{0x0060, 0x0000, 0x0109},
		{0x0000, 0x2000, 0x0110},
		{0x400c, 0x1400, 0x0131},
		{0x200a, 0x2100, 0x0139},
		{0x030c, 0x2200, 0x0141},
		{0x0000, 0x0100, 0x0178},
		{0x0000, 0xc000, 0x0180},
		{0x0000, 0x0000, 0x0181},
		{0x0004, 0x0000, 0x0188},
		{0x0002, 0x0000, 0x0189},
		{0x0000, 0x4140, 0x0190},
		{0x0000, 0x4000, 0x0198},
		{0x400c, 0x1200, 0x01b9},
		{0x002c, 0x0300, 0x01c1},
		{0x0000, 0x0000, 0x01c2},
		{0x0010, 0x0150, 0x01c3},
		{0x0064, 0x0000, 0x01c4},
		{0x400c, 0x0a00, 0x01c9},
		{0x010c, 0x1200, 0x01d1},
		{0x300a, 0x1300, 0x01d9},
		{0x0010, 0x0000, 0x0200},
		{0x0000, 0x0001, 0x0208},
		{0x0004, 0x0000, 0x0209},
		{0x0000, 0x80c0, 0x0210},
		{0x0008, 0x0000, 0x0211},
		{0x0000, 0x3000, 0x0218},
		{0x0008, 0x0000, 0x0219},
		{0x0200, 0x3000, 0x0220},
		{0x0002, 0x0000, 0x0221},
		{0x0000, 0x9000, 0x0228},
		{0x0000, 0x0000, 0x0229},
		{0x0000, 0x3fc0, 0x0248},
		{0x000a, 0x2000, 0x0249},
		{0x200a, 0x0a00, 0x0251},
		{0x0000, 0x3fc0, 0x0258},
		{0x000a, 0x0300, 0x0259},
		{0x0000, 0x3fc0, 0x0260},
		{0x000a, 0x2000, 0x0261},
		{0x400a, 0x2300, 0x0269},
		{0x0000, 0x3fc0, 0x0270},
		{0x000a, 0x2000, 0x0271},
		{0x0004, 0x0000, 0x0290},
		{0x0000, 0x0003, 0x0298},
		{0x0010, 0x0001, 0x02a0},
		{0x0000, 0x0000, 0x02a8},
		{0x0020, 0x0000, 0x02b0},
		{0x0002, 0x0000, 0x02b8},
		{0x0000, 0x0001, 0x02c0},
		{0xc000, 0x1400, 0x02d8},
		{0xfa00, 0x00cf, 0x02d9},
		{0x0000, 0x0000, 0x02da},
		{0x0000, 0x0002, 0x02db},
		{0xc801, 0x1400, 0x02e0},
		{0xfa00, 0x00cf, 0x02e1},
		{0x0000, 0x0000, 0x02e2},
		{0x0000, 0x0002, 0x02e3},
		{0xd87d, 0x1412, 0x02e8},
		{0x6a40, 0x00cf, 0x02e9},
		{0x0000, 0x0000, 0x02ea},
		{0x0000, 0x0002, 0x02eb},
		{0xd004, 0x1400, 0x02f0},
		{0xfa00, 0x00cf, 0x02f1},
		{0x0000, 0x0000, 0x02f2},
		{0x0000, 0x0002, 0x02f3},
		{0xd04b, 0x1412, 0x02f8},
		{0x6a40, 0x00cf, 0x02f9},
		{0x0000, 0x0000, 0x02fa},
		{0x0000, 0x0002, 0x02fb},
		{0xc005, 0x1400, 0x0300},
		{0xfa00, 0x00cf, 0x0301},
		{0x0000, 0x0000, 0x0302},
		{0x0000, 0x0002, 0x0303},
		{0xc819, 0x1412, 0x0308},
		{0x6a40, 0x00cf, 0x0309},
		{0x0000, 0x0000, 0x030a},
		{0x0000, 0x0002, 0x030b},
	};


/// kernel_gemm_0
volatile unsigned short cin_kernel_gemm_0[12][3] __attribute__((aligned(8))) = {
		{0x2800, 0x6400, 0x0018},
		{0x0000, 0x0000, 0x0019},
		{0x0000, 0x0000, 0x001a},
		{0x0000, 0x08c2, 0x001b},
		{0x0020, 0x0000, 0x001c},
		{0x2000, 0x6400, 0x0020},
		{0x0000, 0x0000, 0x0021},
		{0x0000, 0x0000, 0x0022},
		{0x0000, 0x0002, 0x0023},
		{0x0001, 0x0000, 0x0068},
		{0x999a, 0x3f99, 0x00a8},
		{0x000a, 0x0200, 0x00a9},
	};


void gemm(void* arg_0 ,void* arg_1 ,void* arg_2){
  long long unsigned t0, t1, t2, t3, t4, t5;
  for (int int_3 = 0; int_3 < 20; int_3 = int_3 + 1){
    {
    /// %0 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
    uint64_t dramoffset_0 = 100 * int_3;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 100, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 100;
    
    }
    {
    /// kernel_gemm_0
    load_cfg((void*)cin_kernel_gemm_0, 0x20000, 72, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 12, _task_id, 0);
    execute(0xc, _task_id, EX_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %1, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
    uint64_t dramoffset_1 = 100 * int_3;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    store(arg_0 + dramoffset_1 + roffset_1, 0x2000 + spadoffset_1, 100, _task_id, 0);
    spadoffset_1 = spadoffset_1 + 100;
    
    }
    _task_id++;
	// fence(1);
	// t0 = rdcycle();
    {
    /// %2 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
    uint64_t dramoffset_2 = 100 * int_3;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    load_data(arg_0 + dramoffset_2 + roffset_2, 0x0 + spadoffset_2, 100, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_2 = spadoffset_2 + 100;
    // fence(1);
	// t1 = rdcycle();
    }
    {
    /// %3 = ADORA.BlockLoad %arg1 [%arg3, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
    uint64_t dramoffset_3 = 120 * int_3;
    uint64_t spadoffset_3 = 0;
    uint64_t roffset_3 = 0;
    load_data(arg_1 + dramoffset_3 + roffset_3, 0x10000 + spadoffset_3, 120, 1, _task_id, LD_DEP_ST_LAST_TASK);
    load_data(arg_1 + dramoffset_3 + roffset_3, 0x12000 + spadoffset_3, 120, 1, _task_id, LD_DEP_ST_LAST_TASK);
    load_data(arg_1 + dramoffset_3 + roffset_3, 0x8000 + spadoffset_3, 120, 1, _task_id, LD_DEP_ST_LAST_TASK);
    load_data(arg_1 + dramoffset_3 + roffset_3, 0x2000 + spadoffset_3, 120, 1, _task_id, LD_DEP_ST_LAST_TASK);
    load_data(arg_1 + dramoffset_3 + roffset_3, 0x14000 + spadoffset_3, 120, 1, _task_id, LD_DEP_ST_LAST_TASK);
    load_data(arg_1 + dramoffset_3 + roffset_3, 0x18000 + spadoffset_3, 120, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_3 = spadoffset_3 + 120;
    // fence(1);
	// t2 = rdcycle();
    }
    {
    /// %4 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "4", KernelName = "kernel_gemm_1"}
    uint64_t dramoffset_4 = 0;
    uint64_t spadoffset_4 = 0;
    uint64_t roffset_4 = 0;
    load_data(arg_2 + dramoffset_4 + roffset_4, 0xa000 + spadoffset_4, 3000, 1, _task_id, LD_DEP_ST_LAST_TASK);
    load_data(arg_2 + dramoffset_4 + roffset_4, 0x1a000 + spadoffset_4, 3000, 1, _task_id, LD_DEP_ST_LAST_TASK);
    load_data(arg_2 + dramoffset_4 + roffset_4, 0xc000 + spadoffset_4, 3000, 1, _task_id, LD_DEP_ST_LAST_TASK);
    load_data(arg_2 + dramoffset_4 + roffset_4, 0x1c000 + spadoffset_4, 3000, 1, _task_id, LD_DEP_ST_LAST_TASK);
    load_data(arg_2 + dramoffset_4 + roffset_4, 0x4000 + spadoffset_4, 3000, 1, _task_id, LD_DEP_ST_LAST_TASK);
    load_data(arg_2 + dramoffset_4 + roffset_4, 0x16000 + spadoffset_4, 3000, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_4 = spadoffset_4 + 3000;
    // fence(1);
	// t3 = rdcycle();    
    }
    {
    /// kernel_gemm_1
    load_cfg((void*)cin_kernel_gemm_1, 0x20000, 732, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 122, _task_id, 0);
    // fence(1);
	// t4 = rdcycle();  
    execute(0x7f7f, _task_id, EX_DEP_ST_LAST_TASK);
    // fence(1);
	// t5 = rdcycle();  

	// printf("ld: %ld, ld: %ld, ld: %ld, cfg: %ld, exe: %ld.\n", t1-t0, t2-t1, t3-t2, t4-t3, t5-t4);
  
    }
    {
    /// ADORA.BlockStore %5, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "5", KernelName = "kernel_gemm_1"}
    uint64_t dramoffset_5 = 100 * int_3;
    uint64_t spadoffset_5 = 0;
    uint64_t roffset_5 = 0;
    store(arg_0 + dramoffset_5 + roffset_5, 0x6000 + spadoffset_5, 100, _task_id, 0);
    spadoffset_5 = spadoffset_5 + 100;
    
    }
    _task_id++;
  }
  
  
  
}
