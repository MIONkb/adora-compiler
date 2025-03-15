module {
  func.func @kernel_correlation_kernel_1(%arg0: memref<?xf32>, %arg1: memref<?x1200xf32>, %arg2: memref<?xf32>) attributes {Kernel, kernel_correlation_kernel_1} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.400000e+03 : f32
    %cst_1 = arith.constant 1.000000e-01 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    affine.for %arg3 = 0 to 1200 {
      affine.store %cst, %arg0[%arg3] : memref<?xf32>
      %0 = ADORA.BlockLoad %arg1 [0, %arg3] : memref<?x1200xf32> -> memref<1400x1xf32>  {Id = "0", KernelName = "kernel_correlation_kernel_1"}
      %1 = ADORA.BlockLoad %arg2 [%arg3] : memref<?xf32> -> memref<1xf32>  {Id = "1", KernelName = "kernel_correlation_kernel_1"}
      %2 = ADORA.BlockLoad %arg0 [%arg3] : memref<?xf32> -> memref<1xf32>  {Id = "2", KernelName = "kernel_correlation_kernel_1"}
      %3 = ADORA.LocalMemAlloc memref<1xf32>  {Id = "3", KernelName = "kernel_correlation_kernel_1"}
      ADORA.kernel {
        affine.for %arg4 = 0 to 1400 {
          %9 = affine.load %0[%arg4, 0] : memref<1400x1xf32>
          %10 = affine.load %1[0] : memref<1xf32>
          %11 = arith.subf %9, %10 : f32
          %12 = arith.mulf %11, %11 : f32
          %13 = affine.load %2[0] : memref<1xf32>
          %14 = arith.addf %13, %12 : f32
          affine.store %14, %3[0] : memref<1xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_correlation_kernel_1"}
      ADORA.BlockStore %3, %arg0 [%arg3] : memref<1xf32> -> memref<?xf32>  {Id = "3", KernelName = "kernel_correlation_kernel_1"}
      %4 = affine.load %arg0[%arg3] : memref<?xf32>
      %5 = arith.divf %4, %cst_0 : f32
      %6 = math.sqrt %5 : f32
      %7 = arith.cmpf ole, %6, %cst_1 : f32
      %8 = arith.select %7, %cst_2, %6 : f32
      affine.store %8, %arg0[%arg3] : memref<?xf32>
    }
    return
  }
}

