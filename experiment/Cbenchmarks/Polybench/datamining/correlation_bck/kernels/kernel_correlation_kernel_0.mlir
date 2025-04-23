func.func @kernel_correlation_kernel_0(%arg0: memref<?xf32>, %arg1: memref<?x1200xf32>) attributes {Kernel, kernel_correlation_kernel_0} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.400000e+03 : f32
  affine.for %arg2 = 0 to 1200 {
    affine.store %cst, %arg0[%arg2] : memref<?xf32>
    affine.for %arg3 = 0 to 1400 {
      %2 = affine.load %arg1[%arg3, %arg2] : memref<?x1200xf32>
      %3 = affine.load %arg0[%arg2] : memref<?xf32>
      %4 = arith.addf %3, %2 : f32
      affine.store %4, %arg0[%arg2] : memref<?xf32>
    }
    %0 = affine.load %arg0[%arg2] : memref<?xf32>
    %1 = arith.divf %0, %cst_0 : f32
    affine.store %1, %arg0[%arg2] : memref<?xf32>
  }
  return
}