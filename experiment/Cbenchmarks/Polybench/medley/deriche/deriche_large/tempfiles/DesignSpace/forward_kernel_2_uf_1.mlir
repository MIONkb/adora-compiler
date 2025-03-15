module {
  func.func @forward_kernel_2(%arg0: memref<1x64x112x112xf32>, %arg1: memref<1x64x112x112xf32>) attributes {Kernel, forward_kernel_2} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 112 step 16 {
          %0 = ADORA.BlockLoad %arg0 [0, %arg3, %arg4, 0] : memref<1x64x112x112xf32> -> memref<1x1x16x112xf32>  {Id = "0", KernelName = "forward_kernel_2"}
          %1 = ADORA.LocalMemAlloc memref<1x1x16x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
          ADORA.kernel {
            affine.for %arg5 = 0 to 16 {
              affine.for %arg6 = 0 to 112 {
                %2 = affine.load %0[0, 0, %arg5, %arg6] : memref<1x1x16x112xf32>
                %3 = arith.cmpf ugt, %2, %cst : f32
                %4 = arith.select %3, %2, %cst : f32
                affine.store %4, %1[0, 0, %arg5, %arg6] : memref<1x1x16x112xf32>
              }
            }
            ADORA.terminator
          } {KernelName = "forward_kernel_2"}
          ADORA.BlockStore %1, %arg1 [0, %arg3, %arg4, 0] : memref<1x1x16x112xf32> -> memref<1x64x112x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
        }
      }
    }
    return
  }
}
