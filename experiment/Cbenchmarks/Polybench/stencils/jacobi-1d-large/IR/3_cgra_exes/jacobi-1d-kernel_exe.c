
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void jacobi_1d(void* arg_0 ,void* arg_1){
  for (int int_2 = 0; int_2 < 500; int_2 = int_2 + 1){
    {
    /// %0 = ADORA.BlockLoad %arg0 [0] : memref<2000xf32> -> memref<2000xf32>  {Id = "0", KernelName = "kernel_jacobi_1d_0"}
    uint64_t dramoffset_0 = 0;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x10000 + spadoffset_0, 8000, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 8000;
    
    }
    {
    /// %1 = ADORA.BlockLoad %arg0 [1] : memref<2000xf32> -> memref<2000xf32>  {Id = "1", KernelName = "kernel_jacobi_1d_0"}
    uint64_t dramoffset_1 = 0;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_0 + dramoffset_1 + roffset_1, 0x18000 + spadoffset_1, 8000, 0, 0, 0);
    spadoffset_1 = spadoffset_1 + 8000;
    
    }
    {
    /// %2 = ADORA.BlockLoad %arg0 [2] : memref<2000xf32> -> memref<2000xf32>  {Id = "2", KernelName = "kernel_jacobi_1d_0"}
    uint64_t dramoffset_2 = 0;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    load_data(arg_0 + dramoffset_2 + roffset_2, 0x12000 + spadoffset_2, 8000, 0, 0, 0);
    spadoffset_2 = spadoffset_2 + 8000;
    
    }
    {
    /// kernel_jacobi_1d_0
    volatile unsigned short cin[24][3] __attribute__((aligned(8))) = {
    		{0x008b, 0x0118, 0x0259},
    		{0x100b, 0x0118, 0x0261},
    		{0xaa3b, 0x3eaa, 0x0268},
    		{0x0008, 0x0018, 0x0269},
    		{0x0010, 0x0000, 0x02a0},
    		{0x0004, 0x0000, 0x02a8},
    		{0x0102, 0x0003, 0x02b0},
    		{0x0000, 0x0001, 0x02b8},
    		{0x2000, 0x3800, 0x02e0},
    		{0x001f, 0x0000, 0x02e1},
    		{0x0000, 0x0100, 0x02e2},
    		{0x0000, 0x0000, 0x02e3},
    		{0x2000, 0x3800, 0x02f0},
    		{0x001f, 0x0000, 0x02f1},
    		{0x0000, 0x0100, 0x02f2},
    		{0x0000, 0x0000, 0x02f3},
    		{0x2800, 0x3800, 0x02f8},
    		{0x001f, 0x0000, 0x02f9},
    		{0x0000, 0x8100, 0x02fa},
    		{0x0004, 0x0000, 0x02fb},
    		{0x2000, 0x3800, 0x0300},
    		{0x001f, 0x0000, 0x0301},
    		{0x0000, 0x0100, 0x0302},
    		{0x0000, 0x0000, 0x0303},
    	};
    
    load_cfg((void*)cin, 0x20000, 144, 0, 0);
    config(0x0, 24, 0, 0);
    execute(0x3e20, 0, 0);
    }
    {
    /// ADORA.BlockStore %3, %arg1 [1] : memref<2000xf32> -> memref<2000xf32>  {Id = "3", KernelName = "kernel_jacobi_1d_0"}
    uint64_t dramoffset_3 = 0;
    uint64_t spadoffset_3 = 0;
    uint64_t roffset_3 = 0;
    store(arg_1 + dramoffset_3 + roffset_3, 0x1a000 + spadoffset_3, 8000, 0, 0);
    spadoffset_3 = spadoffset_3 + 8000;
    
    }
    {
    /// %4 = ADORA.BlockLoad %arg1 [0] : memref<2000xf32> -> memref<2000xf32>  {Id = "0", KernelName = "kernel_jacobi_1d_1"}
    uint64_t dramoffset_0 = 0;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_1 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 8000, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 8000;
    
    }
    {
    /// %5 = ADORA.BlockLoad %arg1 [1] : memref<2000xf32> -> memref<2000xf32>  {Id = "1", KernelName = "kernel_jacobi_1d_1"}
    uint64_t dramoffset_1 = 0;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_1 + dramoffset_1 + roffset_1, 0x18000 + spadoffset_1, 8000, 0, 0, 0);
    spadoffset_1 = spadoffset_1 + 8000;
    
    }
    {
    /// %6 = ADORA.BlockLoad %arg1 [2] : memref<2000xf32> -> memref<2000xf32>  {Id = "2", KernelName = "kernel_jacobi_1d_1"}
    uint64_t dramoffset_2 = 0;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    load_data(arg_1 + dramoffset_2 + roffset_2, 0x10000 + spadoffset_2, 8000, 0, 0, 0);
    spadoffset_2 = spadoffset_2 + 8000;
    
    }
    {
    /// kernel_jacobi_1d_1
    volatile unsigned short cin[28][3] __attribute__((aligned(8))) = {
    		{0x2000, 0x3800, 0x0030},
    		{0x001f, 0x0000, 0x0031},
    		{0x0000, 0x0100, 0x0032},
    		{0x0000, 0x0000, 0x0033},
    		{0x0000, 0x0010, 0x0078},
    		{0x0003, 0x0000, 0x0109},
    		{0x2000, 0x0004, 0x0198},
    		{0x100b, 0x00d0, 0x01d9},
    		{0x000b, 0x00c8, 0x01e1},
    		{0x0200, 0x0000, 0x0220},
    		{0x0200, 0x0000, 0x0228},
    		{0xaa3b, 0x3eaa, 0x0260},
    		{0x0008, 0x0010, 0x0261},
    		{0x0000, 0x0000, 0x02a8},
    		{0x0000, 0x0004, 0x02b0},
    		{0x0000, 0x0000, 0x02b8},
    		{0x2800, 0x3800, 0x02e8},
    		{0x001f, 0x0000, 0x02e9},
    		{0x0000, 0x8100, 0x02ea},
    		{0x1004, 0x0000, 0x02eb},
    		{0x2000, 0x3800, 0x02f0},
    		{0x001f, 0x0000, 0x02f1},
    		{0x0000, 0x0100, 0x02f2},
    		{0x0000, 0x0000, 0x02f3},
    		{0x2000, 0x3800, 0x0300},
    		{0x001f, 0x0000, 0x0301},
    		{0x0000, 0x0100, 0x0302},
    		{0x0000, 0x0000, 0x0303},
    	};
    
    load_cfg((void*)cin, 0x20000, 168, 0, 0);
    config(0x0, 28, 0, 0);
    execute(0x3e20, 0, 0);
    }
    {
    /// ADORA.BlockStore %7, %arg0 [1] : memref<2000xf32> -> memref<2000xf32>  {Id = "3", KernelName = "kernel_jacobi_1d_1"}
    uint64_t dramoffset_3 = 0;
    uint64_t spadoffset_3 = 0;
    uint64_t roffset_3 = 0;
    store(arg_0 + dramoffset_3 + roffset_3, 0x12000 + spadoffset_3, 8000, 0, 0);
    spadoffset_3 = spadoffset_3 + 8000;
    
    }
  }
  
  
  
}
