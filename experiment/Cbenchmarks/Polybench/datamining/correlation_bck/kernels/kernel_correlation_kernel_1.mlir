func.func @kernel_correlation_kernel_1(%arg0: memref<?xf32>, %arg1: memref<?x1200xf32>, %arg2: memref<?xf32>) attributes {Kernel, kernel_correlation_kernel_1} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.400000e+03 : f32
  %cst_1 = arith.constant 1.000000e-01 : f32
  %cst_2 = arith.constant 1.000000e+00 : f32
  affine.for %arg3 = 0 to 1200 {
    affine.store %cst, %arg0[%arg3] : memref<?xf32>
    affine.for %arg4 = 0 to 1400 {
      %5 = affine.load %arg1[%arg4, %arg3] : memref<?x1200xf32>
      %6 = affine.load %arg2[%arg3] : memref<?xf32>
      %7 = arith.subf %5, %6 : f32
      %8 = arith.mulf %7, %7 : f32
      %9 = affine.load %arg0[%arg3] : memref<?xf32>
      %10 = arith.addf %9, %8 : f32
      affine.store %10, %arg0[%arg3] : memref<?xf32>
    }
    %0 = affine.load %arg0[%arg3] : memref<?xf32>
    %1 = arith.divf %0, %cst_0 : f32
    %2 = math.sqrt %1 : f32
    %3 = arith.cmpf ole, %2, %cst_1 : f32
    %4 = arith.select %3, %cst_2, %2 : f32
    affine.store %4, %arg0[%arg3] : memref<?xf32>
  }
  return
}