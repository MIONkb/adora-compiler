func.func @forward_kernel_1(%arg0: memref<64x64xf32>, %arg1: memref<64xf32>, %arg2: memref<64x64xf32>) attributes {Kernel, forward_kernel_1} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  affine.for %arg3 = 0 to 64 {
    affine.for %arg4 = 0 to 64 {
      %0 = affine.load %arg0[%arg3, %arg4] : memref<64x64xf32>
      %1 = affine.load %arg1[%arg4] : memref<64xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %arg2[%arg3, %arg4] : memref<64x64xf32>
    }
  }
  return
}