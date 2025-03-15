
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void deriche_kernel_2(void* arg_0 ,void* arg_1 ,void* arg_2){
    for (int int_3 = 0; int_3 < 4096; int_3 = int_3 + 1){
      /// %0 = ADORA.BlockLoad %arg0 [%arg3, %c0] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "0", KernelName = "deriche_kernel_2"}
      uint64_t dramoffset_0 = 8640 * int_3;
      uint64_t spadoffset_0 = 0;
      for(int idx_0 = 0; idx_0 < 1; idx_0++){
        uint64_t roffset_0 =  8640*idx_0 ;
        load_data(arg_0 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 4320, 0, 0, 0);
        spadoffset_0 = spadoffset_0 + 4320;
      } 
      /// %1 = ADORA.BlockLoad %arg1 [%arg3, %c0] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "1", KernelName = "deriche_kernel_2"}
      uint64_t dramoffset_1 = 8640 * int_3;
      uint64_t spadoffset_1 = 0;
      for(int idx_0 = 0; idx_0 < 1; idx_0++){
        uint64_t roffset_1 =  8640*idx_0 ;
        load_data(arg_1 + dramoffset_1 + roffset_1, 0xa000 + spadoffset_1, 4320, 0, 0, 0);
        spadoffset_1 = spadoffset_1 + 4320;
      } 
      /// %3 = ADORA.BlockLoad %arg0 [%arg3, %c1080] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "3", KernelName = "deriche_kernel_2"}
      uint64_t dramoffset_3 = 8640 * int_3 + 4 * 1080;
      uint64_t spadoffset_3 = 0;
      for(int idx_0 = 0; idx_0 < 1; idx_0++){
        uint64_t roffset_3 =  8640*idx_0 ;
        load_data(arg_0 + dramoffset_3 + roffset_3, 0x0 + spadoffset_3, 4320, 0, 0, 0);
        spadoffset_3 = spadoffset_3 + 4320;
      } 
      /// %4 = ADORA.BlockLoad %arg1 [%arg3, %c1080] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "4", KernelName = "deriche_kernel_2"}
      uint64_t dramoffset_4 = 8640 * int_3 + 4 * 1080;
      uint64_t spadoffset_4 = 0;
      for(int idx_0 = 0; idx_0 < 1; idx_0++){
        uint64_t roffset_4 =  8640*idx_0 ;
        load_data(arg_1 + dramoffset_4 + roffset_4, 0x2000 + spadoffset_4, 4320, 0, 0, 0);
        spadoffset_4 = spadoffset_4 + 4320;
      } 
      volatile unsigned short cin[30][3] __attribute__((aligned(8))) = {
      		{0x2800, 0xe000, 0x0008},
      		{0x0010, 0x0000, 0x0009},
      		{0x0000, 0x0100, 0x000a},
      		{0x0000, 0x0000, 0x000b},
      		{0x2000, 0xe000, 0x0010},
      		{0x0010, 0x0000, 0x0011},
      		{0x0000, 0x0100, 0x0012},
      		{0x0000, 0x0000, 0x0013},
      		{0x3000, 0xe000, 0x0018},
      		{0x0010, 0x0000, 0x0019},
      		{0x0000, 0x8900, 0x001a},
      		{0x0000, 0x0000, 0x001b},
      		{0x2000, 0xe000, 0x0028},
      		{0x0010, 0x0000, 0x0029},
      		{0x0000, 0x0100, 0x002a},
      		{0x0000, 0x0000, 0x002b},
      		{0x2800, 0xe000, 0x0030},
      		{0x0010, 0x0000, 0x0031},
      		{0x0000, 0x0100, 0x0032},
      		{0x0000, 0x0000, 0x0033},
      		{0x3000, 0xe000, 0x0038},
      		{0x0010, 0x0000, 0x0039},
      		{0x0000, 0x8900, 0x003a},
      		{0x0000, 0x0000, 0x003b},
      		{0x0200, 0x0000, 0x0058},
      		{0x0010, 0x0000, 0x0060},
      		{0x0200, 0x0000, 0x0078},
      		{0x0010, 0x0000, 0x0080},
      		{0x000e, 0x0012, 0x00a1},
      		{0x000e, 0x0012, 0x00c1},
      	};
      
      load_cfg((void*)cin, 0x20000, 180, 0, 0);
      config(0x0, 30, 0, 0);
      execute(0x77, 0, 0);
      /// ADORA.BlockStore %2, %arg2 [%arg3, %c0] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "2", KernelName = "deriche_kernel_2"}
      uint64_t dramoffset_2 = 4320 * int_3;
      uint64_t spadoffset_2 = 0;
      uint64_t roffset_2 = 0;
      store(arg_2 + dramoffset_2 + roffset_2, 0xc000 + spadoffset_2, 4320, 0, 0);
      spadoffset_2 = spadoffset_2 + 4320;
      
      /// ADORA.BlockStore %5, %arg2 [%arg3, %c1080_0] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "5", KernelName = "deriche_kernel_2"}
      uint64_t dramoffset_5 = 4320 * int_3 + 4 * 1080;
      uint64_t spadoffset_5 = 0;
      uint64_t roffset_5 = 0;
      store(arg_2 + dramoffset_5 + roffset_5, 0x4000 + spadoffset_5, 4320, 0, 0);
      spadoffset_5 = spadoffset_5 + 4320;
      
      }
}
