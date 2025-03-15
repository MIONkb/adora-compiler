module {
  func.func @forward_kernel_0(%arg0: memref<64x128xf32>, %arg1: memref<128x64xf32>, %arg2: memref<64x64xf32>) attributes {Kernel, forward_kernel_0} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 step 8 {
        %0 = ADORA.BlockLoad %arg2 [%arg3, %arg4] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "0", KernelName = "forward_kernel_0"}
        %1 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "1", KernelName = "forward_kernel_0"}
        %2 = ADORA.BlockLoad %arg1 [0, %arg4] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "2", KernelName = "forward_kernel_0"}
        %3 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "3", KernelName = "forward_kernel_0"}
        ADORA.kernel {
          affine.for %arg5 = 0 to 8 {
            %4 = affine.load %0[0, %arg5] : memref<1x8xf32>
            %5 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %4) -> (f32) {
              %6 = affine.load %1[0, %arg6] : memref<1x128xf32>
              %7 = affine.load %2[%arg6, %arg5] : memref<128x8xf32>
              %8 = arith.mulf %6, %7 : f32
              %9 = arith.addf %arg7, %8 : f32
              affine.yield %9 : f32
            }
            affine.store %5, %3[0, %arg5] : memref<1x8xf32>
          }
          ADORA.terminator
        } {KernelName = "forward_kernel_0"}
        ADORA.BlockStore %3, %arg2 [%arg3, %arg4] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      }
    }
    return
  }
}

