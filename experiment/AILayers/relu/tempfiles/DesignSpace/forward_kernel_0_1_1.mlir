module {
  func.func @forward_kernel_0(%arg0: memref<1x128x64xf32>, %arg1: memref<1x128x64xf32>) attributes {Kernel, forward_kernel_0} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 step 16 {
        %0 = ADORA.BlockLoad %arg0 [0, %arg3, 0] : memref<1x128x64xf32> -> memref<1x16x64xf32>  {Id = "0", KernelName = "forward_kernel_0"}
        %1 = ADORA.LocalMemAlloc memref<1x16x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
        ADORA.kernel {
          affine.for %arg4 = 0 to 16 {
            affine.for %arg5 = 0 to 64 {
              %2 = affine.load %0[0, %arg4, %arg5] : memref<1x16x64xf32>
              %3 = arith.cmpf ugt, %2, %cst : f32
              %4 = arith.select %3, %2, %cst : f32
              affine.store %4, %1[0, %arg4, %arg5] : memref<1x16x64xf32>
            }
          }
          ADORA.terminator
        } {KernelName = "forward_kernel_0"}
        ADORA.BlockStore %1, %arg1 [0, %arg3, 0] : memref<1x16x64xf32> -> memref<1x128x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      }
    }
    return
  }
}
