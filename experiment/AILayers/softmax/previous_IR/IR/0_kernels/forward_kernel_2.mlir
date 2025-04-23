func.func @forward_kernel_2(%arg0: memref<1x128x64xf32>, %arg1: memref<1x128x64xf32>) attributes {Kernel, forward_kernel_2} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 128 {
      affine.for %arg4 = 0 to 64 {
        %0 = affine.load %arg0[0, %arg3, %arg4] : memref<1x128x64xf32>
        %1 = math.rsqrt %0 : f32
        affine.store %1, %arg1[%arg2, %arg3, %arg4] : memref<1x128x64xf32>
      }
    }
  }
  return
}