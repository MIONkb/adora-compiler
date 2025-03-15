func.func @kernel_correlation_kernel_2(%arg0: memref<?xf32>, %arg1: memref<?x1200xf32>, %arg2: memref<?xf32>) attributes {Kernel, kernel_correlation_kernel_2} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  %cst = arith.constant 37.4165726 : f32
  affine.for %arg3 = 0 to 1400 {
    affine.for %arg4 = 0 to 1200 {
      %0 = affine.load %arg0[%arg4] : memref<?xf32>
      %1 = affine.load %arg1[%arg3, %arg4] : memref<?x1200xf32>
      %2 = arith.subf %1, %0 : f32
      // affine.store %2, %arg1[%arg3, %arg4] : memref<?x1200xf32>
      %3 = affine.load %arg2[%arg4] : memref<?xf32>
      %4 = arith.mulf %3, %cst : f32
      %5 = arith.divf %2, %4 : f32
      affine.store %5, %arg1[%arg3, %arg4] : memref<?x1200xf32>
    }
  }
  return
}