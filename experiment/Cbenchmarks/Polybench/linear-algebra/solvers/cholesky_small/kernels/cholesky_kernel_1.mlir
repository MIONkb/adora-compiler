func.func @cholesky_kernel_1(%arg0: memref<120x120xf32>, %arg1: index) attributes {Kernel, cholesky_kernel_1} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  affine.for %arg2 = 0 to affine_map<(d0) -> (d0)>(%arg1) {
    %0 = affine.load %arg0[%arg1, %arg2] : memref<120x120xf32>
    %1 = arith.mulf %0, %0 : f32
    %2 = affine.load %arg0[%arg1, %arg1] : memref<120x120xf32>
    %3 = arith.subf %2, %1 : f32
    affine.store %3, %arg0[%arg1, %arg1] : memref<120x120xf32>
  }
  return
}