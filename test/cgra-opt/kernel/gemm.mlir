module attributes {} {
  func.func @gemm_opt(%arg0: memref<?x25xf32>, %arg1: memref<?x30xf32>, %arg2: memref<?x25xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.200000e+00 : f32
    %cst_0 = arith.constant 1.500000e+00 : f32
    affine.for %arg3 = 0 to 20 {
      affine.for %arg4 = 0 to 25 {
        %0 = affine.load %arg0[%arg3, %arg4] : memref<?x25xf32>
        %1 = arith.mulf %0, %cst : f32
        affine.store %1, %arg0[%arg3, %arg4] : memref<?x25xf32>
      }
    }
    affine.for %arg3 = 0 to 20 {
      affine.for %arg4 = 0 to 25 {
        affine.for %arg5 = 0 to 30 {
          %0 = affine.load %arg1[%arg3, %arg5] : memref<?x30xf32>
          %1 = arith.mulf %0, %cst_0 : f32
          %2 = affine.load %arg2[%arg5, %arg4] : memref<?x25xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = affine.load %arg0[%arg3, %arg4] : memref<?x25xf32>
          %5 = arith.addf %4, %3 : f32
          affine.store %5, %arg0[%arg3, %arg4] : memref<?x25xf32>
        }
      }
    }
    return
  }
}

