module {
  func.func @forward_kernel_1(%arg0: memref<64x64xf32>, %arg1: memref<64xf32>, %arg2: memref<64x64xf32>) attributes {Kernel, forward_kernel_1} {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c48 = arith.constant 48 : index
    %c0 = arith.constant 0 : index
    %0 = ADORA.BlockLoad %arg0 [%c0, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "0", KernelName = "forward_kernel_1"}
    %1 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "1", KernelName = "forward_kernel_1"}
    %2 = ADORA.LocalMemAlloc memref<16x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
    %3 = ADORA.BlockLoad %arg0 [%c16, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "3", KernelName = "forward_kernel_1"}
    %4 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "4", KernelName = "forward_kernel_1"}
    %5 = ADORA.LocalMemAlloc memref<16x64xf32>  {Id = "5", KernelName = "forward_kernel_1"}
    %6 = ADORA.BlockLoad %arg0 [%c32, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "6", KernelName = "forward_kernel_1"}
    %7 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "7", KernelName = "forward_kernel_1"}
    %8 = ADORA.LocalMemAlloc memref<16x64xf32>  {Id = "8", KernelName = "forward_kernel_1"}
    %9 = ADORA.BlockLoad %arg0 [%c48, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "9", KernelName = "forward_kernel_1"}
    %10 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "10", KernelName = "forward_kernel_1"}
    %11 = ADORA.LocalMemAlloc memref<16x64xf32>  {Id = "11", KernelName = "forward_kernel_1"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 64 {
          %12 = affine.load %0[%arg3, %arg4] : memref<16x64xf32>
          %13 = affine.load %1[%arg4] : memref<64xf32>
          %14 = arith.addf %12, %13 : f32
          affine.store %14, %2[%arg3, %arg4] : memref<16x64xf32>
          %15 = affine.load %3[%arg3, %arg4] : memref<16x64xf32>
          %16 = affine.load %4[%arg4] : memref<64xf32>
          %17 = arith.addf %15, %16 : f32
          affine.store %17, %5[%arg3, %arg4] : memref<16x64xf32>
          %18 = affine.load %6[%arg3, %arg4] : memref<16x64xf32>
          %19 = affine.load %7[%arg4] : memref<64xf32>
          %20 = arith.addf %18, %19 : f32
          affine.store %20, %8[%arg3, %arg4] : memref<16x64xf32>
          %21 = affine.load %9[%arg3, %arg4] : memref<16x64xf32>
          %22 = affine.load %10[%arg4] : memref<64xf32>
          %23 = arith.addf %21, %22 : f32
          affine.store %23, %11[%arg3, %arg4] : memref<16x64xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "forward_kernel_1"}
    ADORA.BlockStore %2, %arg2 [%c0, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
    ADORA.BlockStore %5, %arg2 [%c16, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "5", KernelName = "forward_kernel_1"}
    ADORA.BlockStore %8, %arg2 [%c32, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "8", KernelName = "forward_kernel_1"}
    ADORA.BlockStore %11, %arg2 [%c48, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "11", KernelName = "forward_kernel_1"}
    return
  }
}