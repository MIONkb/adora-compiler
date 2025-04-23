module {
  func.func @kernel_correlation_kernel_0(%arg0: memref<?xf32>, %arg1: memref<?x1200xf32>) attributes {Kernel, kernel_correlation_kernel_0} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.400000e+03 : f32
    affine.for %arg2 = 0 to 1200 {
      affine.store %cst, %arg0[%arg2] : memref<?xf32>
      %0 = ADORA.BlockLoad %arg1 [0, %arg2] : memref<?x1200xf32> -> memref<1400x1xf32>  {Id = "0", KernelName = "kernel_correlation_kernel_0"}
      %1 = ADORA.BlockLoad %arg0 [%arg2] : memref<?xf32> -> memref<1xf32>  {Id = "1", KernelName = "kernel_correlation_kernel_0"}
      %2 = ADORA.LocalMemAlloc memref<1xf32>  {Id = "2", KernelName = "kernel_correlation_kernel_0"}
      ADORA.kernel {
        affine.for %arg3 = 0 to 1400 {
          %5 = affine.load %0[%arg3, 0] : memref<1400x1xf32>
          %6 = affine.load %1[0] : memref<1xf32>
          %7 = arith.addf %6, %5 : f32
          affine.store %7, %2[0] : memref<1xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_correlation_kernel_0"}
      ADORA.BlockStore %2, %arg0 [%arg2] : memref<1xf32> -> memref<?xf32>  {Id = "2", KernelName = "kernel_correlation_kernel_0"}
      %3 = affine.load %arg0[%arg2] : memref<?xf32>
      %4 = arith.divf %3, %cst_0 : f32
      affine.store %4, %arg0[%arg2] : memref<?xf32>
    }
    return
  }
}

