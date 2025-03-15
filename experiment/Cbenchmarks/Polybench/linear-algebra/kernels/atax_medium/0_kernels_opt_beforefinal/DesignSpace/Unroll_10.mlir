module {
  func.func @atax_kernel_0(%arg0: memref<390x410xf32>, %arg1: index, %arg2: memref<410xf32>, %arg3: memref<390xf32>) attributes {Kernel, atax_kernel_0} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    %0 = ADORA.BlockLoad %arg0 [%arg1, 0] : memref<390x410xf32> -> memref<1x410xf32>  {Id = "0", KernelName = ""}
    %1 = ADORA.BlockLoad %arg2 [0] : memref<410xf32> -> memref<410xf32>  {Id = "1", KernelName = ""}
    %2 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "2", KernelName = ""}
    ADORA.kernel {
      %3 = affine.for %arg4 = 0 to 410 step 10 iter_args(%arg5 = %cst) -> (f32) {
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
        %24 = affine.load %0[0, %arg4 + 5] : memref<1x410xf32>
        %25 = affine.load %1[%arg4 + 5] : memref<410xf32>
        %26 = arith.mulf %24, %25 : f32
        %27 = arith.addf %23, %26 : f32
        %28 = affine.load %0[0, %arg4 + 6] : memref<1x410xf32>
        %29 = affine.load %1[%arg4 + 6] : memref<410xf32>
        %30 = arith.mulf %28, %29 : f32
        %31 = arith.addf %27, %30 : f32
        %32 = affine.load %0[0, %arg4 + 7] : memref<1x410xf32>
        %33 = affine.load %1[%arg4 + 7] : memref<410xf32>
        %34 = arith.mulf %32, %33 : f32
        %35 = arith.addf %31, %34 : f32
        %36 = affine.load %0[0, %arg4 + 8] : memref<1x410xf32>
        %37 = affine.load %1[%arg4 + 8] : memref<410xf32>
        %38 = arith.mulf %36, %37 : f32
        %39 = arith.addf %35, %38 : f32
        %40 = affine.load %0[0, %arg4 + 9] : memref<1x410xf32>
        %41 = affine.load %1[%arg4 + 9] : memref<410xf32>
        %42 = arith.mulf %40, %41 : f32
        %43 = arith.addf %39, %42 : f32
        affine.yield %43 : f32
      }
      affine.store %3, %2[0] : memref<2xf32>
      ADORA.terminator
    }
    ADORA.BlockStore %2, %arg3 [%arg1] : memref<2xf32> -> memref<390xf32>  {Id = "2", KernelName = ""}
    return
  }
}
