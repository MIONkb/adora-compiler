func.func @forward_kernel_3(%arg0: memref<1x128x64xf32>, %arg1: memref<1x128x1xf32>) attributes {Kernel, forward_kernel_3} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 128 {
      affine.for %arg4 = 0 to 64 {
        %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<1x128x64xf32>
        %1 = affine.load %arg1[%arg2, %arg3, 0] : memref<1x128x1xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %arg1[%arg2, %arg3, 0] : memref<1x128x1xf32>
      }
    }
  }
  return
}