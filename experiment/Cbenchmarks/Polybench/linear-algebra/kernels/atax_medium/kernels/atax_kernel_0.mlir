func.func @atax_kernel_0(%arg0: memref<390x410xf32>, %arg1: index, %arg2: memref<410xf32>, %arg3: memref<390xf32>) attributes {Kernel, atax_kernel_0} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  %cst = arith.constant 0.000000e+00 : f32
  ADORA.kernel{
  %0 = affine.for %arg4 = 0 to 410 iter_args(%arg5 = %cst) -> (f32) {
    %1 = affine.load %arg0[%arg1, %arg4] : memref<390x410xf32>
    %2 = affine.load %arg2[%arg4] : memref<410xf32>
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %arg5, %3 : f32
    affine.yield %4 : f32
  }
  affine.store %0, %arg3[%arg1] : memref<390xf32>
  ADORA.terminator
  }
  return
}