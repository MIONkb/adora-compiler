#include "include/ISA.h"
// #include "cgrv_test.h"
#include "include/encoding.h"

void cholesky_kernel_0(void* arg_0 ,int arg_1 ,int arg_2){
    /// %0 = ADORA.BlockLoad %arg0 [%arg1, 0] : memref<120x120xf32> -> memref<1x120xf32>  {Id = "0", KernelName = "cholesky_kernel_0"}
    printf("Start kernel 0\n");
    uint64_t dramoffset_0 = 480 * arg_1;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 480, 0, 0, 0);
    spadoffset_0 = spadoffset_0 + 480;
    printf("Finish first load\n");
    
    /// %1 = ADORA.BlockLoad %arg0 [%arg2, 0] : memref<120x120xf32> -> memref<1x120xf32>  {Id = "1", KernelName = "cholesky_kernel_0"}
    uint64_t dramoffset_1 = 480 * arg_2;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    load_data(arg_0 + dramoffset_1 + roffset_1, 0x0 + spadoffset_1, 480, 0, 0, 0);
    spadoffset_1 = spadoffset_1 + 480;
    printf("Finish second load\n");
    
    /// %2 = ADORA.BlockLoad %arg0 [%arg1, %arg2] : memref<120x120xf32> -> memref<1x2xf32>  {Id = "2", KernelName = "cholesky_kernel_0"}
    uint64_t dramoffset_2 = 480 * arg_1 + 4 * arg_2;
    uint64_t spadoffset_2 = 0;
    for(int idx_0 = 0; idx_0 < 1; idx_0++){
      uint64_t roffset_2 =  480*idx_0 ;
      load_data(arg_0 + dramoffset_2 + roffset_2, 0x2000 + spadoffset_2, 8, 0, 0, 0);
      spadoffset_2 = spadoffset_2 + 8;
    } 

    printf("Finish third load\n");
    int int_3 = arg_2;
    int int_4 = arg_2;
    int int_5 = arg_2;
    int int_6 = arg_2;
    int int_7 = arg_2;
    volatile unsigned short cin[28][3] __attribute__((aligned(8))) = {
    		{0x0800, 0xf800, 0x0010},
    		{0x0013, 0x0000, 0x0011},
    		{0x0000, 0x0100, 0x0012},
    		{0x0000, 0x0000, 0x0013},
    		{0x0000, 0xf800, 0x0018},
    		{0x0013, 0x0000, 0x0019},
    		{0x0000, 0x9100, 0x001a},
    		{0x0200, 0x0000, 0x001b},
    		{0x2000, 0xf800, 0x0020},
    		{0x0013, 0x0000, 0x0021},
    		{0x0000, 0x0100, 0x0022},
    		{0x0000, 0x0000, 0x0023},
    		{0x2000, 0xf800, 0x0040},
    		{0x0013, 0x0000, 0x0041},
    		{0x0000, 0x0100, 0x0042},
    		{0x0000, 0x0000, 0x0043},
    		{0x0000, 0x0000, 0x0060},
    		{0x2001, 0x0000, 0x0068},
    		{0x0000, 0x0001, 0x0070},
    		{0x1400, 0x0000, 0x0078},
    		{0x2000, 0x0000, 0x0080},
    		{0x0000, 0x0000, 0x0088},
    		{0x080e, 0x0014, 0x00a9},
    		{0x0010, 0x0004, 0x00b9},
    		{0x0000, 0x0400, 0x00ba},
    		{0xe180, 0x014f, 0x00bb},
    		{0x0000, 0x0000, 0x00bc},
    		{0x020d, 0x0014, 0x00c1},
    	};
    
    cin[12][1] = (int_3 << 0xa) | (cin[12][1] & 0x3ff);
    cin[13][0] = (int_3 >> 0x6) | (cin[13][0] & 0xfc00);
    cin[8][1] = (int_4 << 0xa) | (cin[8][1] & 0x3ff);
    cin[9][0] = (int_4 >> 0x6) | (cin[9][0] & 0xfc00);
    cin[0][1] = (int_5 << 0xa) | (cin[0][1] & 0x3ff);
    cin[1][0] = (int_5 >> 0x6) | (cin[1][0] & 0xfc00);
    cin[4][1] = (int_6 << 0xa) | (cin[4][1] & 0x3ff);
    cin[5][0] = (int_6 >> 0x6) | (cin[5][0] & 0xfc00);
    cin[25][0] = (int_7 << 0xc) | (cin[25][0] & 0xfff);
    cin[25][1] = (int_7 >> 0x8) | (cin[25][1] & 0xff00);
    load_cfg((void*)cin, 0x20000, 168, 0, 0);
    printf("Finish load cfg\n");

    config(0x0, 28, 0, 0);
    printf("Finish cfg\n");
    
    execute(0x8a, 0, 0);
    printf("Finish execute\n");

    /// ADORA.BlockStore %3, %arg0 [%arg1, %arg2] : memref<1x2xf32> -> memref<120x120xf32>  {Id = "3", KernelName = "cholesky_kernel_0"}
    uint64_t dramoffset_3 = 8 * arg_1 + 4 * arg_2;
    uint64_t spadoffset_3 = 0;
    uint64_t roffset_3 = 0;
    store(arg_0 + dramoffset_3 + roffset_3, 0x0 + spadoffset_3, 8, 0, 0);
    spadoffset_3 = spadoffset_3 + 8;
    printf("Finish store\n");
}