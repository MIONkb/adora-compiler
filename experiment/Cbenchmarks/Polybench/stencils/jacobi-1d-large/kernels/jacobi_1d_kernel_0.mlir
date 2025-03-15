func.func @jacobi_1d_kernel_0(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {Kernel, jacobi_1d_kernel_0} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  %cst = arith.constant 3.333300e-01 : f32
  affine.for %arg2 = 0 to 1998 {
    %0 = affine.load %arg0[%arg2] : memref<2000xf32>
    %1 = affine.load %arg0[%arg2 + 1] : memref<2000xf32>
    %2 = arith.addf %0, %1 : f32
    %3 = affine.load %arg0[%arg2 + 2] : memref<2000xf32>
    %4 = arith.addf %2, %3 : f32
    %5 = arith.mulf %4, %cst : f32
    affine.store %5, %arg1[%arg2 + 1] : memref<2000xf32>
  }
  return
}