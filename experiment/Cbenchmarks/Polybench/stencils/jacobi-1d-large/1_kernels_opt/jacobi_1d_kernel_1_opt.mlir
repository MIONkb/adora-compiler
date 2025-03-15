module {
  func.func @jacobi_1d_kernel_1(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {Kernel, jacobi_1d_kernel_1} {
    %c999 = arith.constant 999 : index
    %cst = arith.constant 3.333300e-01 : f32
    %c0 = arith.constant 0 : index
    %0 = ADORA.BlockLoad %arg0 [%c0] : memref<2000xf32> -> memref<1000xf32>  {Id = "0", KernelName = "jacobi_1d_kernel_1"}
    %1 = ADORA.BlockLoad %arg0 [%c0 + 1] : memref<2000xf32> -> memref<1000xf32>  {Id = "1", KernelName = "jacobi_1d_kernel_1"}
    %2 = ADORA.BlockLoad %arg0 [%c0 + 2] : memref<2000xf32> -> memref<1000xf32>  {Id = "2", KernelName = "jacobi_1d_kernel_1"}
    %3 = ADORA.LocalMemAlloc memref<1000xf32>  {Id = "3", KernelName = "jacobi_1d_kernel_1"}
    %4 = ADORA.BlockLoad %arg0 [%c999] : memref<2000xf32> -> memref<1000xf32>  {Id = "4", KernelName = "jacobi_1d_kernel_1"}
    %5 = ADORA.BlockLoad %arg0 [%c999 + 1] : memref<2000xf32> -> memref<1000xf32>  {Id = "5", KernelName = "jacobi_1d_kernel_1"}
    %6 = ADORA.BlockLoad %arg0 [%c999 + 2] : memref<2000xf32> -> memref<1000xf32>  {Id = "6", KernelName = "jacobi_1d_kernel_1"}
    %7 = ADORA.LocalMemAlloc memref<1000xf32>  {Id = "7", KernelName = "jacobi_1d_kernel_1"}
    ADORA.kernel {
      affine.for %arg2 = 0 to 999 {
        %8 = affine.load %0[%arg2] : memref<1000xf32>
        %9 = affine.load %1[%arg2 + 1] : memref<1000xf32>
        %10 = arith.addf %8, %9 : f32
        %11 = affine.load %2[%arg2 + 2] : memref<1000xf32>
        %12 = arith.addf %10, %11 : f32
        %13 = arith.mulf %12, %cst : f32
        affine.store %13, %3[%arg2 + 1] : memref<1000xf32>
        %14 = affine.load %4[%arg2] : memref<1000xf32>
        %15 = affine.load %5[%arg2 + 1] : memref<1000xf32>
        %16 = arith.addf %14, %15 : f32
        %17 = affine.load %6[%arg2 + 2] : memref<1000xf32>
        %18 = arith.addf %16, %17 : f32
        %19 = arith.mulf %18, %cst : f32
        affine.store %19, %7[%arg2 + 1] : memref<1000xf32>
      }
      ADORA.terminator
    } {KernelName = "jacobi_1d_kernel_1"}
    ADORA.BlockStore %3, %arg1 [%c0 + 1] : memref<1000xf32> -> memref<2000xf32>  {Id = "3", KernelName = "jacobi_1d_kernel_1"}
    ADORA.BlockStore %7, %arg1 [%c999 + 1] : memref<1000xf32> -> memref<2000xf32>  {Id = "7", KernelName = "jacobi_1d_kernel_1"}
    return
  }
}