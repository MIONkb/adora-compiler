func.func @kernel_correlation_kernel_3(%arg0: memref<?x1200xf32>, %arg1: memref<?x1200xf32>) attributes {Kernel, kernel_correlation_kernel_3} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  affine.for %arg2 = 0 to 1199 {
    affine.store %cst, %arg0[%arg2, %arg2] : memref<?x1200xf32>
    affine.for %arg3 = 0 to affine_map<(d0) -> (-d0 + 1199)>(%arg2) {
      affine.store %cst_0, %arg0[%arg2, %arg2 + %arg3 + 1] : memref<?x1200xf32>
      affine.for %arg4 = 0 to 1400 {
        %1 = affine.load %arg1[%arg4, %arg2] : memref<?x1200xf32>
        %2 = affine.load %arg1[%arg4, %arg2 + %arg3 + 1] : memref<?x1200xf32>
        %3 = arith.mulf %1, %2 : f32
        %4 = affine.load %arg0[%arg2, %arg2 + %arg3 + 1] : memref<?x1200xf32>
        %5 = arith.addf %4, %3 : f32
        affine.store %5, %arg0[%arg2, %arg2 + %arg3 + 1] : memref<?x1200xf32>
      }
      %0 = affine.load %arg0[%arg2, %arg2 + %arg3 + 1] : memref<?x1200xf32>
      affine.store %0, %arg0[%arg2 + %arg3 + 1, %arg2] : memref<?x1200xf32>
    }
  }
  return
}