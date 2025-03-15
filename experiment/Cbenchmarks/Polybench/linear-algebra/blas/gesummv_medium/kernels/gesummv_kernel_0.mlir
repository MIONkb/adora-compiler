func.func @gesummv_kernel_0(%arg0: memref<250xf32>, %arg1: memref<250xf32>, %arg2: memref<250x250xf32>, %arg3: memref<250xf32>, %arg4: memref<250x250xf32>) attributes {Kernel, gesummv_kernel_0} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.500000e+00 : f32
  %cst_1 = arith.constant 1.200000e+00 : f32
  affine.for %arg5 = 0 to 250 {
    affine.store %cst, %arg0[%arg5] : memref<250xf32>
    affine.store %cst, %arg1[%arg5] : memref<250xf32>
    affine.for %arg6 = 0 to 250 {
      %5 = affine.load %arg2[%arg5, %arg6] : memref<250x250xf32>
      %6 = affine.load %arg3[%arg6] : memref<250xf32>
      %7 = arith.mulf %5, %6 : f32
      %8 = affine.load %arg0[%arg5] : memref<250xf32>
      %9 = arith.addf %7, %8 : f32
      affine.store %9, %arg0[%arg5] : memref<250xf32>
      %10 = affine.load %arg4[%arg5, %arg6] : memref<250x250xf32>
      // %11 = affine.load %arg3[%arg6] : memref<250xf32>
      %12 = arith.mulf %10, %6 : f32
      %13 = affine.load %arg1[%arg5] : memref<250xf32>
      %14 = arith.addf %12, %13 : f32
      affine.store %14, %arg1[%arg5] : memref<250xf32>
    }
    %0 = affine.load %arg0[%arg5] : memref<250xf32>
    %1 = arith.mulf %0, %cst_0 : f32
    %2 = affine.load %arg1[%arg5] : memref<250xf32>
    %3 = arith.mulf %2, %cst_1 : f32
    %4 = arith.addf %1, %3 : f32
    affine.store %4, %arg1[%arg5] : memref<250xf32>
  }
  return
}