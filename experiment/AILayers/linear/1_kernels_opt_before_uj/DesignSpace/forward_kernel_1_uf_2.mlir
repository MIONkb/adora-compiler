#map = affine_map<(d0) -> (d0 + 16)>
module {
  func.func @forward_kernel_1(%arg0: memref<64x64xf32>, %arg1: memref<64xf32>, %arg2: memref<64x64xf32>) attributes {Kernel, forward_kernel_1} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    affine.for %arg3 = 0 to 64 step 32 {
      %0 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "0", KernelName = "forward_kernel_1"}
      %1 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "1", KernelName = "forward_kernel_1"}
      %2 = ADORA.LocalMemAlloc memref<16x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
      %3 = affine.apply #map(%arg3)
      %4 = ADORA.BlockLoad %arg0 [%3, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "0", KernelName = "forward_kernel_1"}
      %5 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "1", KernelName = "forward_kernel_1"}
      %6 = ADORA.LocalMemAlloc memref<16x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
      ADORA.kernel {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 64 {
            %8 = affine.load %0[%arg4, %arg5] : memref<16x64xf32>
            %9 = affine.load %1[%arg5] : memref<64xf32>
            %10 = arith.addf %8, %9 : f32
            affine.store %10, %2[%arg4, %arg5] : memref<16x64xf32>
            %11 = affine.apply #map(%arg3)
            %12 = affine.load %4[%arg4, %arg5] : memref<16x64xf32>
            %13 = affine.load %5[%arg5] : memref<64xf32>
            %14 = arith.addf %12, %13 : f32
            affine.store %14, %6[%arg4, %arg5] : memref<16x64xf32>
          }
        }
        ADORA.terminator
      } {KernelName = "forward_kernel_1"}
      ADORA.BlockStore %2, %arg2 [%arg3, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
      %7 = affine.apply #map(%arg3)
      ADORA.BlockStore %6, %arg2 [%7, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
    }
    return
  }
}
