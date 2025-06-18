module {
  func.func @forward_kernel_0(%arg0: memref<64x128xf32>, %arg1: memref<128x64xf32>, %arg2: memref<64x64xf32>) attributes {Kernel, forward_kernel_0} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 {
        affine.for %arg5 = 0 to 128 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<64x128xf32>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<128x64xf32>
          %2 = affine.load %arg2[%arg3, %arg4] : memref<64x64xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %arg2[%arg3, %arg4] : memref<64x64xf32>
        }
      }
    }
    return
  }
}

