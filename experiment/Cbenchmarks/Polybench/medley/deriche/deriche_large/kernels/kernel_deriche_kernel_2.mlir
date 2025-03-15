func.func @kernel_deriche_kernel_2(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>) attributes {Kernel, kernel_deriche_kernel_2} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  affine.for %arg3 = 0 to 4096 {
    affine.for %arg4 = 0 to 2160 {
      %0 = affine.load %arg0[%arg3, %arg4] : memref<4096x2160xf32>
      %1 = affine.load %arg1[%arg3, %arg4] : memref<4096x2160xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %arg2[%arg3, %arg4] : memref<4096x2160xf32>
    }
  }
  return
}