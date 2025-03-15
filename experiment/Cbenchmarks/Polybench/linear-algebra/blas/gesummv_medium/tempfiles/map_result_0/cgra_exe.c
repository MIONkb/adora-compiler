
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

void gesummv_kernel_0(void* arg_0 ,void* arg_1 ,void* arg_2 ,void* arg_3 ,void* arg_4){
    for (int int_5 = 0; int_5 < 250; int_5 = int_5 + 5){
      /// %0 = ADORA.BlockLoad %arg2 [%arg5, 0] : memref<250x250xf32> -> memref<5x250xf32>  {Id = "0", KernelName = "gesummv_kernel_0"}
      uint64_t dramoffset_0 = 1000 * int_5;
      uint64_t spadoffset_0 = 0;
      uint64_t roffset_0 = 0;
      load_data(arg_2 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 5000, 1, 0, 0);
      load_data(arg_2 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 5000, 0, 0, 0);
      spadoffset_0 = spadoffset_0 + 5000;
      
      /// %1 = ADORA.BlockLoad %arg3 [0] : memref<250xf32> -> memref<250xf32>  {Id = "1", KernelName = "gesummv_kernel_0"}
      uint64_t dramoffset_1 = 0;
      uint64_t spadoffset_1 = 0;
      uint64_t roffset_1 = 0;
      load_data(arg_3 + dramoffset_1 + roffset_1, 0xa000 + spadoffset_1, 1000, 1, 0, 0);
      load_data(arg_3 + dramoffset_1 + roffset_1, 0x10000 + spadoffset_1, 1000, 0, 0, 0);
      spadoffset_1 = spadoffset_1 + 1000;
      
      /// %2 = ADORA.BlockLoad %arg4 [%arg5, 0] : memref<250x250xf32> -> memref<5x250xf32>  {Id = "2", KernelName = "gesummv_kernel_0"}
      uint64_t dramoffset_2 = 1000 * int_5;
      uint64_t spadoffset_2 = 0;
      uint64_t roffset_2 = 0;
      load_data(arg_4 + dramoffset_2 + roffset_2, 0x18000 + spadoffset_2, 5000, 1, 0, 0);
      load_data(arg_4 + dramoffset_2 + roffset_2, 0x12000 + spadoffset_2, 5000, 0, 0, 0);
      spadoffset_2 = spadoffset_2 + 5000;
      
      volatile unsigned short cin[78][3] __attribute__((aligned(8))) = {
      		{0x4000, 0xf400, 0x0008},
      		{0x0081, 0x0028, 0x0009},
      		{0x0000, 0x0100, 0x000a},
      		{0x0000, 0x0000, 0x000b},
      		{0x0800, 0xf400, 0x0020},
      		{0x0041, 0x0028, 0x0021},
      		{0x0000, 0x9d00, 0x0022},
      		{0x0000, 0x0000, 0x0023},
      		{0x1000, 0xf400, 0x0028},
      		{0x0041, 0x0028, 0x0029},
      		{0x0000, 0x9500, 0x002a},
      		{0x0200, 0x0000, 0x002b},
      		{0x4800, 0xf400, 0x0038},
      		{0xc201, 0x002f, 0x0039},
      		{0x0000, 0x0100, 0x003a},
      		{0x0000, 0x0000, 0x003b},
      		{0x4000, 0xf400, 0x0040},
      		{0x0081, 0x0028, 0x0041},
      		{0x0000, 0x0100, 0x0042},
      		{0x0000, 0x0000, 0x0043},
      		{0x0000, 0x0000, 0x0058},
      		{0x0000, 0x0004, 0x0060},
      		{0x0030, 0x0008, 0x0068},
      		{0x0000, 0x0008, 0x0070},
      		{0x1101, 0x0000, 0x0078},
      		{0x4200, 0x0000, 0x0080},
      		{0x0000, 0x0000, 0x0088},
      		{0x004d, 0x0032, 0x00a1},
      		{0x0010, 0x0004, 0x00b9},
      		{0x0000, 0x0400, 0x00ba},
      		{0xd240, 0x0107, 0x00bb},
      		{0x0000, 0x0000, 0x00bc},
      		{0x060e, 0x0022, 0x00c1},
      		{0x000d, 0x0014, 0x00c9},
      		{0x0200, 0x0000, 0x00e8},
      		{0x0000, 0x0000, 0x00f8},
      		{0x4000, 0x0004, 0x0100},
      		{0x0003, 0x0000, 0x0119},
      		{0x0003, 0x0000, 0x0138},
      		{0x000d, 0x0008, 0x0139},
      		{0x002e, 0x0034, 0x0141},
      		{0x0003, 0x0000, 0x0148},
      		{0x000d, 0x0010, 0x0149},
      		{0x0000, 0x1000, 0x0178},
      		{0x0000, 0x8000, 0x0180},
      		{0x0001, 0x0000, 0x0181},
      		{0x1200, 0x0000, 0x0188},
      		{0xc000, 0x0000, 0x01a8},
      		{0x0010, 0x0002, 0x01d1},
      		{0x0000, 0x0400, 0x01d2},
      		{0xd240, 0x0107, 0x01d3},
      		{0x0000, 0x0000, 0x01d4},
      		{0x000d, 0x0016, 0x01f1},
      		{0x0000, 0x1000, 0x0208},
      		{0x0000, 0x6c00, 0x0210},
      		{0x0000, 0x0880, 0x0218},
      		{0x0000, 0x0030, 0x0220},
      		{0x0000, 0x0180, 0x0228},
      		{0x0000, 0x0180, 0x0230},
      		{0x00c0, 0x0000, 0x0238},
      		{0x000d, 0x0038, 0x0259},
      		{0x004e, 0x0026, 0x0261},
      		{0x0000, 0x0000, 0x0298},
      		{0x0000, 0x0000, 0x02a0},
      		{0x0000, 0x0000, 0x02a8},
      		{0x0000, 0x0000, 0x02c8},
      		{0x4000, 0xf400, 0x02e0},
      		{0xc201, 0x002f, 0x02e1},
      		{0x0000, 0x0100, 0x02e2},
      		{0x0000, 0x0000, 0x02e3},
      		{0x4800, 0xf400, 0x02f0},
      		{0x0081, 0x0028, 0x02f1},
      		{0x0000, 0x0100, 0x02f2},
      		{0x0000, 0x0000, 0x02f3},
      		{0x4000, 0xf400, 0x0310},
      		{0x0081, 0x0028, 0x0311},
      		{0x0000, 0x0100, 0x0312},
      		{0x0000, 0x0000, 0x0313},
      	};
      
      load_cfg((void*)cin, 0x20000, 468, 0, 0);
      config(0x0, 78, 0, 0);
      execute(0x8ad9, 0, 0);
      /// ADORA.BlockStore %4, %arg1 [%arg5] : memref<6xf32> -> memref<250xf32>  {Id = "4", KernelName = "gesummv_kernel_0"}
      uint64_t dramoffset_4 = 4 * int_5;
      uint64_t spadoffset_4 = 0;
      uint64_t roffset_4 = 0;
      store(arg_1 + dramoffset_4 + roffset_4, 0x2000 + spadoffset_4, 24, 0, 0);
      spadoffset_4 = spadoffset_4 + 24;
      
      /// ADORA.BlockStore %3, %arg0 [%arg5] : memref<6xf32> -> memref<250xf32>  {Id = "3", KernelName = "gesummv_kernel_0"}
      uint64_t dramoffset_3 = 4 * int_5;
      uint64_t spadoffset_3 = 0;
      uint64_t roffset_3 = 0;
      store(arg_0 + dramoffset_3 + roffset_3, 0xc000 + spadoffset_3, 24, 0, 0);
      spadoffset_3 = spadoffset_3 + 24;
      
      }
}
