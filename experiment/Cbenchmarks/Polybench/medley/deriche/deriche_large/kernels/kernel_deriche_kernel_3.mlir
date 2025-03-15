func.func @kernel_deriche_kernel_3(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>) attributes {Kernel, kernel_deriche_kernel_3} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant -0.188681662 : f32
  %cst_1 = arith.constant 0.110209078 : f32
  %cst_2 = arith.constant 0.840896427 : f32
  %cst_3 = arith.constant -0.606530666 : f32
  // affine.for %arg5 = 0 to 2160 {
  affine.for %arg6 = 0 to 4096 {
    affine.store %cst, %arg0[] : memref<f32>
    affine.store %cst, %arg1[] : memref<f32>
    affine.store %cst, %arg2[] : memref<f32>
    ADORA.kernel{
    // affine.for %arg6 = 0 to 4096 {
    affine.for %arg5 = 0 to 2160 {
      %0 = affine.load %arg3[%arg6, %arg5] : memref<4096x2160xf32>
      %1 = arith.mulf %0, %cst_0 : f32
      %2 = affine.load %arg0[] : memref<f32>
      %3 = arith.mulf %2, %cst_1 : f32
      %4 = arith.addf %1, %3 : f32
      %5 = affine.load %arg1[] : memref<f32>
      %6 = arith.mulf %5, %cst_2 : f32
      %7 = arith.addf %4, %6 : f32
      %8 = affine.load %arg2[] : memref<f32>
      %9 = arith.mulf %8, %cst_3 : f32
      %10 = arith.addf %7, %9 : f32
      affine.store %10, %arg4[%arg6, %arg5] : memref<4096x2160xf32>
      %11 = affine.load %arg3[%arg6, %arg5] : memref<4096x2160xf32>
      affine.store %11, %arg0[] : memref<f32>
      affine.store %5, %arg2[] : memref<f32>
      %12 = affine.load %arg4[%arg6, %arg5] : memref<4096x2160xf32>
      affine.store %12, %arg1[] : memref<f32>
    }
    ADORA.terminator}
  }
  return
}