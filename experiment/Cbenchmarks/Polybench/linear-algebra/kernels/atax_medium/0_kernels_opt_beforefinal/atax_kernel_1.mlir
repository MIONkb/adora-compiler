module {
  func.func @atax_kernel_1(%arg0: memref<390xf32>, %arg1: index, %arg2: memref<410xf32>, %arg3: memref<390x410xf32>) attributes {Kernel, atax_kernel_1} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %1 = ADORA.BlockLoad %arg2 [0] : memref<410xf32> -> memref<410xf32>  {Id = "0", KernelName = "atax_kernel_1"}
    %2 = ADORA.BlockLoad %arg3 [%arg1, 0] : memref<390x410xf32> -> memref<1x410xf32>  {Id = "1", KernelName = "atax_kernel_1"}
    %a = ADORA.BlockLoad %arg0 [%arg1] : memref<390xf32> -> memref<2xf32>  {Id = "2", KernelName = "atax_kernel_1"}
    %3 = ADORA.LocalMemAlloc memref<410xf32>  {Id = "3", KernelName = "atax_kernel_1"}
    ADORA.kernel {
      affine.for %arg4 = 0 to 390 {
        %0 = affine.load %a[0] : memref<2xf32>
        %4 = affine.load %1[%arg4] : memref<410xf32>
        %5 = affine.load %2[0, %arg4] : memref<1x410xf32>
        %6 = arith.mulf %5, %0 : f32
        %7 = arith.addf %4, %6 : f32
        affine.store %7, %3[%arg4] : memref<410xf32>
      }
      ADORA.terminator
    } {KernelName = "atax_kernel_1"}
    ADORA.BlockStore %3, %arg2 [0] : memref<410xf32> -> memref<410xf32>  {Id = "3", KernelName = "atax_kernel_1"}
    return
  }
}