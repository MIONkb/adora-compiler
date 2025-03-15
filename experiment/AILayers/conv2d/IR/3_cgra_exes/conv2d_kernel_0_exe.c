
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void conv2d_kernel_0(void* arg_0 ,void* arg_1 ,void* arg_2){
    for (int int_3 = 0; int_3 < 1; int_3 = int_3 + 1){
      for (int int_4 = 0; int_4 < 6; int_4 = int_4 + 1){
        for (int int_5 = 0; int_5 < 58; int_5 = int_5 + 1){
          for (int int_6 = 0; int_6 < 58; int_6 = int_6 + 29){
            {
            /// %0 = ADORA.BlockLoad %arg2 [%arg3, %arg4, %arg5, %arg6] : memref<1x6x58x58xf32> -> memref<1x1x1x30xf32>  {Id = "0", KernelName = "conv2d_kernel_0"}
            uint64_t dramoffset_0 = 80736 * int_3 + 13456 * int_4 + 232 * int_5 + 4 * int_6;
            uint64_t spadoffset_0 = 0;
            for(int idx_1 = 0; idx_1 < 1; idx_1++){
              for(int idx_2 = 0; idx_2 < 1; idx_2++){
                uint64_t roffset_0 =  13456*idx_1 + 232*idx_2 ;
                load_data(arg_2 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 120, 0, 0, 0);
                spadoffset_0 = spadoffset_0 + 120;
            } } 
            }
            {
            /// %1 = ADORA.BlockLoad %arg0 [%arg3, 0, %arg5, %arg6] : memref<1x3x64x64xf32> -> memref<1x3x7x36xf32>  {Id = "1", KernelName = "conv2d_kernel_0"}
            uint64_t dramoffset_1 = 49152 * int_3 + 256 * int_5 + 4 * int_6;
            uint64_t spadoffset_1 = 0;
            for(int idx_1 = 0; idx_1 < 3; idx_1++){
              for(int idx_2 = 0; idx_2 < 7; idx_2++){
                uint64_t roffset_1 =  16384*idx_1 + 256*idx_2 ;
                load_data(arg_0 + dramoffset_1 + roffset_1, 0x10000 + spadoffset_1, 144, 1, 0, 0);
                load_data(arg_0 + dramoffset_1 + roffset_1, 0x0 + spadoffset_1, 144, 1, 0, 0);
                load_data(arg_0 + dramoffset_1 + roffset_1, 0xa000 + spadoffset_1, 144, 1, 0, 0);
                load_data(arg_0 + dramoffset_1 + roffset_1, 0x18000 + spadoffset_1, 144, 1, 0, 0);
                load_data(arg_0 + dramoffset_1 + roffset_1, 0x12000 + spadoffset_1, 144, 1, 0, 0);
                load_data(arg_0 + dramoffset_1 + roffset_1, 0x14000 + spadoffset_1, 144, 1, 0, 0);
                load_data(arg_0 + dramoffset_1 + roffset_1, 0x2000 + spadoffset_1, 144, 0, 0, 0);
                spadoffset_1 = spadoffset_1 + 144;
            } } 
            }
            {
            /// %2 = ADORA.BlockLoad %arg1 [%arg4, 0, 0, 0] : memref<6x3x7x7xf32> -> memref<2x3x7x7xf32>  {Id = "2", KernelName = "conv2d_kernel_0"}
            uint64_t dramoffset_2 = 588 * int_4;
            uint64_t spadoffset_2 = 0;
            uint64_t roffset_2 = 0;
            load_data(arg_1 + dramoffset_2 + roffset_2, 0x1a000 + spadoffset_2, 1176, 1, 0, 0);
            load_data(arg_1 + dramoffset_2 + roffset_2, 0xc000 + spadoffset_2, 1176, 1, 0, 0);
            load_data(arg_1 + dramoffset_2 + roffset_2, 0x1c000 + spadoffset_2, 1176, 1, 0, 0);
            load_data(arg_1 + dramoffset_2 + roffset_2, 0x1e000 + spadoffset_2, 1176, 1, 0, 0);
            load_data(arg_1 + dramoffset_2 + roffset_2, 0x16000 + spadoffset_2, 1176, 1, 0, 0);
            load_data(arg_1 + dramoffset_2 + roffset_2, 0xe000 + spadoffset_2, 1176, 1, 0, 0);
            load_data(arg_1 + dramoffset_2 + roffset_2, 0x4000 + spadoffset_2, 1176, 0, 0, 0);
            spadoffset_2 = spadoffset_2 + 1176;
            
            }
            {
            /// conv2d_kernel_0
            volatile unsigned short cin[116][3] __attribute__((aligned(8))) = {
            		{0xf000, 0x1c00, 0x0008},
            		{0x01c0, 0x0018, 0x0009},
            		{0xdfba, 0x0101, 0x000a},
            		{0x0000, 0x0000, 0x000b},
            		{0x8000, 0x1c04, 0x0010},
            		{0x0900, 0x8018, 0x0011},
            		{0xde98, 0x0101, 0x0012},
            		{0x0000, 0x0000, 0x0013},
            		{0x1800, 0x1c00, 0x0018},
            		{0x0000, 0x8018, 0x0019},
            		{0xd000, 0xf101, 0x001a},
            		{0x1004, 0x0000, 0x001b},
            		{0x8800, 0x1c04, 0x0020},
            		{0x0900, 0x8018, 0x0021},
            		{0xde98, 0x0101, 0x0022},
            		{0x0000, 0x0000, 0x0023},
            		{0xf000, 0x1c00, 0x0028},
            		{0x01c0, 0x0018, 0x0029},
            		{0xdfba, 0x0101, 0x002a},
            		{0x0000, 0x0000, 0x002b},
            		{0x8800, 0x1c04, 0x0030},
            		{0x0900, 0x8018, 0x0031},
            		{0xde98, 0x0101, 0x0032},
            		{0x0000, 0x0000, 0x0033},
            		{0xf800, 0x1c00, 0x0038},
            		{0x01c0, 0x0018, 0x0039},
            		{0xdfba, 0x0101, 0x003a},
            		{0x0000, 0x0000, 0x003b},
            		{0x0000, 0x1c00, 0x0040},
            		{0x0000, 0x8018, 0x0041},
            		{0xd000, 0x0101, 0x0042},
            		{0x0000, 0x0000, 0x0043},
            		{0x0000, 0x0000, 0x0058},
            		{0x0400, 0x0000, 0x0060},
            		{0x0102, 0x0000, 0x0068},
            		{0x0000, 0x0031, 0x0070},
            		{0x0000, 0x0002, 0x0078},
            		{0x0000, 0x0000, 0x0080},
            		{0x0000, 0x0010, 0x0088},
            		{0x0088, 0x0050, 0x00a9},
            		{0x0008, 0x0088, 0x00b1},
            		{0x480a, 0x0118, 0x00b9},
            		{0x0004, 0x0000, 0x00f1},
            		{0x0000, 0x0080, 0x00f8},
            		{0x00c0, 0x0000, 0x0100},
            		{0x0003, 0x0000, 0x0101},
            		{0x0008, 0x0000, 0x0108},
            		{0x0000, 0x0180, 0x0110},
            		{0x0003, 0x0000, 0x0111},
            		{0x0000, 0x0100, 0x0118},
            		{0x280a, 0x0060, 0x0141},
            		{0x080a, 0x0058, 0x0149},
            		{0x0000, 0x6000, 0x0180},
            		{0x1000, 0x0000, 0x0188},
            		{0x00c0, 0x0400, 0x0190},
            		{0x0003, 0x0000, 0x0191},
            		{0x0000, 0x0000, 0x0198},
            		{0x0003, 0x0000, 0x01a1},
            		{0x001a, 0x0008, 0x01d1},
            		{0x0000, 0x1000, 0x01d2},
            		{0x4d00, 0x0405, 0x01d3},
            		{0x0000, 0x0000, 0x01d4},
            		{0x180a, 0x00e0, 0x01d9},
            		{0x080a, 0x0108, 0x01e1},
            		{0x180a, 0x0118, 0x01e9},
            		{0x0000, 0x2000, 0x0210},
            		{0x0000, 0x4000, 0x0218},
            		{0x0000, 0x4020, 0x0220},
            		{0x8000, 0x0000, 0x0228},
            		{0x2000, 0x0000, 0x0230},
            		{0x0000, 0x0000, 0x0238},
            		{0x0008, 0x0118, 0x0251},
            		{0x0108, 0x0098, 0x0261},
            		{0x0008, 0x0118, 0x0269},
            		{0x100a, 0x0050, 0x0271},
            		{0x0808, 0x00c8, 0x0279},
            		{0x0008, 0x0118, 0x0281},
            		{0x0010, 0x0000, 0x0298},
            		{0x0000, 0x0010, 0x02a0},
            		{0x0020, 0x0000, 0x02a8},
            		{0x0010, 0x0000, 0x02b0},
            		{0x0002, 0x0000, 0x02b8},
            		{0x0000, 0x0000, 0x02c0},
            		{0x0010, 0x0000, 0x02c8},
            		{0x8800, 0x1c04, 0x02d8},
            		{0x0900, 0x8018, 0x02d9},
            		{0xde98, 0x0101, 0x02da},
            		{0x0000, 0x0000, 0x02db},
            		{0x9000, 0x1c04, 0x02e0},
            		{0x0900, 0x8018, 0x02e1},
            		{0xde98, 0x0101, 0x02e2},
            		{0x0000, 0x0000, 0x02e3},
            		{0xf800, 0x1c00, 0x02e8},
            		{0x01c0, 0x0018, 0x02e9},
            		{0xdfba, 0x0101, 0x02ea},
            		{0x0000, 0x0000, 0x02eb},
            		{0x8000, 0x1c04, 0x02f0},
            		{0x0900, 0x8018, 0x02f1},
            		{0xde98, 0x0101, 0x02f2},
            		{0x0000, 0x0000, 0x02f3},
            		{0xe800, 0x1c00, 0x02f8},
            		{0x01c0, 0x0018, 0x02f9},
            		{0xdfba, 0x0101, 0x02fa},
            		{0x0000, 0x0000, 0x02fb},
            		{0xf000, 0x1c00, 0x0300},
            		{0x01c0, 0x0018, 0x0301},
            		{0xdfba, 0x0101, 0x0302},
            		{0x0000, 0x0000, 0x0303},
            		{0x8000, 0x1c04, 0x0308},
            		{0x0900, 0x8018, 0x0309},
            		{0xde98, 0x0101, 0x030a},
            		{0x0000, 0x0000, 0x030b},
            		{0xf800, 0x1c00, 0x0310},
            		{0x01c0, 0x0018, 0x0311},
            		{0xdfba, 0x0101, 0x0312},
            		{0x0000, 0x0000, 0x0313},
            	};
            
            load_cfg((void*)cin, 0x20000, 696, 0, 0);
            config(0x0, 116, 0, 0);
            execute(0xffff, 0, 0);
            }
            {
            /// ADORA.BlockStore %3, %arg2 [0, %arg4, %arg5, %arg6] : memref<1x1x1x30xf32> -> memref<1x6x58x58xf32>  {Id = "3", KernelName = "conv2d_kernel_0"}
            uint64_t dramoffset_3 = 120 * int_4 + 120 * int_5 + 4 * int_6;
            uint64_t spadoffset_3 = 0;
            uint64_t roffset_3 = 0;
            store(arg_2 + dramoffset_3 + roffset_3, 0x6000 + spadoffset_3, 120, 0, 0);
            spadoffset_3 = spadoffset_3 + 120;
            
            }
          }
          
          
          
        }
        
        
        
      }
      
      
      
    }
    
    
    
}
