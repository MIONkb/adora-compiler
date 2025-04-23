module {
  func.func @kernel_correlation_kernel_2(%arg0: memref<?xf32>, %arg1: memref<?x1200xf32>, %arg2: memref<?xf32>) attributes {Kernel, kernel_correlation_kernel_2} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 37.4165726 : f32
    affine.for %arg3 = 0 to 1400 {
      %0 = ADORA.BlockLoad %arg0 [0] : memref<?xf32> -> memref<1200xf32>  {Id = "0", KernelName = "kernel_correlation_kernel_2"}
      %1 = ADORA.BlockLoad %arg1 [%arg3, 0] : memref<?x1200xf32> -> memref<1x1200xf32>  {Id = "1", KernelName = "kernel_correlation_kernel_2"}
      %2 = ADORA.BlockLoad %arg2 [0] : memref<?xf32> -> memref<1200xf32>  {Id = "2", KernelName = "kernel_correlation_kernel_2"}
      %3 = ADORA.LocalMemAlloc memref<1x1200xf32>  {Id = "3", KernelName = "kernel_correlation_kernel_2"}
      ADORA.kernel {
        affine.for %arg4 = 0 to 1200 {
          %4 = affine.load %0[%arg4] : memref<1200xf32>
          %5 = affine.load %1[0, %arg4] : memref<1x1200xf32>
          %6 = arith.subf %5, %4 : f32
          %7 = affine.load %2[%arg4] : memref<1200xf32>
          %8 = arith.mulf %7, %cst : f32
          %9 = arith.divf %6, %8 : f32
          affine.store %9, %3[0, %arg4] : memref<1x1200xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_correlation_kernel_2"}
      ADORA.BlockStore %3, %arg1 [%arg3, 0] : memref<1x1200xf32> -> memref<?x1200xf32>  {Id = "3", KernelName = "kernel_correlation_kernel_2"}
    }
    return
  }
}

