module {
  func.func @gesummv_kernel_0(%arg0: memref<250xf32>, %arg1: memref<250xf32>, %arg2: memref<250x250xf32>, %arg3: memref<250xf32>, %arg4: memref<250x250xf32>) attributes {Kernel, gesummv_kernel_0} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.500000e+00 : f32
    %cst_1 = arith.constant 1.200000e+00 : f32
    ADORA.kernel {
      affine.for %arg5 = 0 to 250 {
        // affine.store %cst, %arg0[%arg5] : memref<250xf32>
        // affine.store %cst, %arg1[%arg5] : memref<250xf32>
        // %0 = affine.load %arg0[%arg5] : memref<250xf32>
        // %1 = affine.load %arg1[%arg5] : memref<250xf32>
        %2:2 = affine.for %arg6 = 0 to 250 iter_args(%arg7 = %cst, %arg8 = %cst) -> (f32, f32) {
          %8 = affine.load %arg2[%arg5, %arg6] : memref<250x250xf32>
          %9 = affine.load %arg3[%arg6] : memref<250xf32>
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %10, %arg7 : f32
          %12 = affine.load %arg4[%arg5, %arg6] : memref<250x250xf32>
          %13 = arith.mulf %12, %9 : f32
          %14 = arith.addf %13, %arg8 : f32
          affine.yield %11, %14 : f32, f32
        }
        affine.store %2#1, %arg1[%arg5] : memref<250xf32>
        affine.store %2#0, %arg0[%arg5] : memref<250xf32>
        // %3 = affine.load %arg0[%arg5] : memref<250xf32>
        // %4 = arith.mulf %3, %cst_0 : f32
        %4 = arith.mulf %2#0, %cst_0 : f32

        // %5 = affine.load %arg1[%arg5] : memref<250xf32>
        // %6 = arith.mulf %5, %cst_1 : f32
        %6 = arith.mulf %2#1, %cst_1 : f32

        %7 = arith.addf %4, %6 : f32
        affine.store %7, %arg1[%arg5] : memref<250xf32>
      }
      ADORA.terminator
    } {KernelName = "gesummv_kernel_0"}
    return
  }
}