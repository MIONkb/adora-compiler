func.func @atax_kernel_1(%arg0: memref<390xf32>, %arg1: index, %arg2: memref<410xf32>, %arg3: memref<390x410xf32>) attributes {Kernel, atax_kernel_1} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  %0 = affine.load %arg0[%arg1] : memref<390xf32>
  affine.for %arg4 = 0 to 390 {
    %1 = affine.load %arg2[%arg4] : memref<410xf32>
    %2 = affine.load %arg3[%arg1, %arg4] : memref<390x410xf32>
    %3 = arith.mulf %2, %0 : f32
    %4 = arith.addf %1, %3 : f32
    affine.store %4, %arg2[%arg4] : memref<410xf32>
  }
  return
}