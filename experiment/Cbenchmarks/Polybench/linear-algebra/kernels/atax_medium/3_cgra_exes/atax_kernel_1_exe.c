
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void atax_kernel_1(void* arg_0 ,int arg_1 ,void* arg_2 ,void* arg_3){
    /// %0 = ADORA.BlockLoad %arg2 [0] : memref<410xf32> -> memref<410xf32>  {Id = "0", KernelName = "atax_kernel_1"}
    uint64_t dramoffset_0 = 0;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_2 + dramoffset_0 + roffset_0, 0x10000 + spadoffset_0, 1640, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 1640;
    
    // /// %1 = ADORA.BlockLoad %arg3 [%arg1, 0] : memref<390x410xf32> -> memref<1x410xf32>  {Id = "1", KernelName = "atax_kernel_1"}
    // uint64_t dramoffset_1 = 1640 * arg_1;
    // uint64_t spadoffset_1 = 0;
    // uint64_t roffset_1 = 0;
    // load_data(arg_3 + dramoffset_1 + roffset_1, 0x8000 + spadoffset_1, 1640, 0, 0, 0);
    // spadoffset_1 = spadoffset_1 + 1640;
    
    // /// %2 = ADORA.BlockLoad %arg0 [%arg1] : memref<390xf32> -> memref<2xf32>  {Id = "2", KernelName = "atax_kernel_1"}
    // uint64_t dramoffset_2 = 4 * arg_1;
    // uint64_t spadoffset_2 = 0;
    // uint64_t roffset_2 = 0;
    // load_data(arg_0 + dramoffset_2 + roffset_2, 0x12000 + spadoffset_2, 8, 0, 0, 0);
    // spadoffset_2 = spadoffset_2 + 8;
    
    volatile unsigned short cin[26][3] __attribute__((aligned(8))) = {
    		{0x2000, 0x1800, 0x0028},
    		{0x0006, 0x0000, 0x0029},
    		{0x0000, 0x0100, 0x002a},
    		{0x0000, 0x0000, 0x002b},
    		{0x0000, 0x0010, 0x0070},
    		{0x0003, 0x0000, 0x0101},
    		{0x0000, 0x0008, 0x0190},
    		{0x000d, 0x0034, 0x01d1},
    		{0x0000, 0x0030, 0x0210},
    		{0x0200, 0x0000, 0x0218},
    		{0x006e, 0x0026, 0x0251},
    		{0x0010, 0x0000, 0x0298},
    		{0x0000, 0x0000, 0x02a0},
    		{0x0000, 0x0000, 0x02a8},
    		{0x2000, 0x1800, 0x02d8},
    		{0x0006, 0x0000, 0x02d9},
    		{0x0000, 0x0100, 0x02da},
    		{0x0000, 0x0000, 0x02db},
    		{0x3000, 0x1800, 0x02e8},
    		{0x0006, 0x0000, 0x02e9},
    		{0x0000, 0x8f00, 0x02ea},
    		{0x0000, 0x0000, 0x02eb},
    		{0x0800, 0x1800, 0x02f0},
    		{0x0006, 0x0000, 0x02f1},
    		{0x0000, 0x0100, 0x02f2},
    		{0x0000, 0x0000, 0x02f3},
    	};
    
    load_cfg((void*)cin, 0x20000, 156, 0, 0);
    config(0x0, 26, 0, 0);
    execute(0xd10, 0, 0);
    /// ADORA.BlockStore %3, %arg2 [0] : memref<410xf32> -> memref<410xf32>  {Id = "3", KernelName = "atax_kernel_1"}
    uint64_t dramoffset_3 = 0;
    uint64_t spadoffset_3 = 0;
    uint64_t roffset_3 = 0;
    store(arg_2 + dramoffset_3 + roffset_3, 0x14000 + spadoffset_3, 1640, 0, 0);
    spadoffset_3 = spadoffset_3 + 1640;
    
}
