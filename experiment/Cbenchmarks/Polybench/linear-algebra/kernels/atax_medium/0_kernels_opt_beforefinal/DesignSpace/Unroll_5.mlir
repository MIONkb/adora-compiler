module {
  func.func @atax_kernel_0(%arg0: memref<390x410xf32>, %arg1: index, %arg2: memref<410xf32>, %arg3: memref<390xf32>) attributes {Kernel, atax_kernel_0} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    %0 = ADORA.BlockLoad %arg0 [%arg1, 0] : memref<390x410xf32> -> memref<1x410xf32>  {Id = "0", KernelName = ""}
    %1 = ADORA.BlockLoad %arg2 [0] : memref<410xf32> -> memref<410xf32>  {Id = "1", KernelName = ""}
    %2 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "2", KernelName = ""}
    ADORA.kernel {
      %3 = affine.for %arg4 = 0 to 410 step 5 iter_args(%arg5 = %cst) -> (f32) {
        %4 = affine.load %0[0, %arg4] : memref<1x410xf32>
        %5 = affine.load %1[%arg4] : memref<410xf32>
        %6 = arith.mulf %4, %5 : f32
        %7 = arith.addf %arg5, %6 : f32
        %8 = affine.load %0[0, %arg4 + 1] : memref<1x410xf32>
        %9 = affine.load %1[%arg4 + 1] : memref<410xf32>
        %10 = arith.mulf %8, %9 : f32
        %11 = arith.addf %7, %10 : f32
        %12 = affine.load %0[0, %arg4 + 2] : memref<1x410xf32>
        %13 = affine.load %1[%arg4 + 2] : memref<410xf32>
        %14 = arith.mulf %12, %13 : f32
        %15 = arith.addf %11, %14 : f32
        %16 = affine.load %0[0, %arg4 + 3] : memref<1x410xf32>
        %17 = affine.load %1[%arg4 + 3] : memref<410xf32>
        %18 = arith.mulf %16, %17 : f32
        %19 = arith.addf %15, %18 : f32
        %20 = affine.load %0[0, %arg4 + 4] : memref<1x410xf32>
        %21 = affine.load %1[%arg4 + 4] : memref<410xf32>
        %22 = arith.mulf %20, %21 : f32
        %23 = arith.addf %19, %22 : f32
        affine.yield %23 : f32
      }
      affine.store %3, %2[0] : memref<2xf32>
      ADORA.terminator
    }
    ADORA.BlockStore %2, %arg3 [%arg1] : memref<2xf32> -> memref<390xf32>  {Id = "2", KernelName = ""}
    return
  }
}
