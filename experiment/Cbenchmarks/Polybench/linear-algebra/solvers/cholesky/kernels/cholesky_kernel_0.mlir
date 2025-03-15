func.func @cholesky_kernel_0(%arg0: memref<?x2000xf32>, %arg1: index, %arg2: index) attributes {Kernel, cholesky_kernel_0} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  affine.for %arg3 = 0 to affine_map<(d0) -> (d0)>(%arg2) {
    %0 = affine.load %arg0[%arg1, %arg3] : memref<?x2000xf32>
    %1 = affine.load %arg0[%arg2, %arg3] : memref<?x2000xf32>
    %2 = arith.mulf %0, %1 : f32
    %3 = affine.load %arg0[%arg1, %arg2] : memref<?x2000xf32>
    %4 = arith.subf %3, %2 : f32
    affine.store %4, %arg0[%arg1, %arg2] : memref<?x2000xf32>
  }
  return
}