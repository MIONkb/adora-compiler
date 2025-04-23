func.func @forward_kernel_0(%arg0: memref<1x128x64xf32>, %arg1: memref<1x128x1xf32>, %arg2: memref<1x128x1xi64>) attributes {Kernel, forward_kernel_0} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  affine.for %arg3 = 0 to 1 {
    affine.for %arg4 = 0 to 128 {
      affine.for %arg5 = 0 to 64 {
        %0 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<1x128x64xf32>
        %1 = affine.load %arg1[%arg3, %arg4, 0] : memref<1x128x1xf32>
        %2 = affine.load %arg2[%arg3, %arg4, 0] : memref<1x128x1xi64>
        %3 = arith.index_cast %arg5 : index to i64
        %4 = arith.cmpf ugt, %0, %1 : f32
        %5 = arith.select %4, %0, %1 : f32
        %6 = arith.cmpf uno, %1, %1 : f32
        %7 = arith.select %6, %1, %5 : f32
        %8 = arith.cmpf ogt, %0, %1 : f32
        %9 = arith.select %8, %3, %2 : i64
        affine.store %7, %arg1[%arg3, %arg4, 0] : memref<1x128x1xf32>
        affine.store %9, %arg2[%arg3, %arg4, 0] : memref<1x128x1xi64>
      }
    }
  }
  return
}