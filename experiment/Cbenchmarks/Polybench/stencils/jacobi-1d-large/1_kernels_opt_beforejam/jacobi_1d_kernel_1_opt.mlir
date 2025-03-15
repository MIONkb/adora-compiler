module {
  func.func @jacobi_1d_kernel_1(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {Kernel, jacobi_1d_kernel_1} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 3.333300e-01 : f32
    affine.for %arg2 = 0 to 1998 step 999 {
      %0 = ADORA.BlockLoad %arg0 [%arg2] : memref<2000xf32> -> memref<1000xf32>  {Id = "0", KernelName = "jacobi_1d_kernel_1"}
      %1 = ADORA.BlockLoad %arg0 [%arg2 + 1] : memref<2000xf32> -> memref<1000xf32>  {Id = "1", KernelName = "jacobi_1d_kernel_1"}
      %2 = ADORA.BlockLoad %arg0 [%arg2 + 2] : memref<2000xf32> -> memref<1000xf32>  {Id = "2", KernelName = "jacobi_1d_kernel_1"}
      %3 = ADORA.LocalMemAlloc memref<1000xf32>  {Id = "3", KernelName = "jacobi_1d_kernel_1"}
      ADORA.kernel {
        affine.for %arg3 = 0 to 999 {
          %4 = affine.load %0[%arg3] : memref<1000xf32>
          %5 = affine.load %1[%arg3 + 1] : memref<1000xf32>
          %6 = arith.addf %4, %5 : f32
          %7 = affine.load %2[%arg3 + 2] : memref<1000xf32>
          %8 = arith.addf %6, %7 : f32
          %9 = arith.mulf %8, %cst : f32
          affine.store %9, %3[%arg3 + 1] : memref<1000xf32>
        }
        ADORA.terminator
      } {KernelName = "jacobi_1d_kernel_1"}
      ADORA.BlockStore %3, %arg1 [%arg2 + 1] : memref<1000xf32> -> memref<2000xf32>  {Id = "3", KernelName = "jacobi_1d_kernel_1"}
    }
    return
  }
}