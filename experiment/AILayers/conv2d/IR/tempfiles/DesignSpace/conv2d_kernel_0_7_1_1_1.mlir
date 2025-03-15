module {
  func.func @conv2d_kernel_0(%arg0: memref<1x3x64x64xf32>, %arg1: memref<6x3x7x7xf32>, %arg2: memref<1x6x58x58xf32>) attributes {Kernel, conv2d_kernel_0} {
    %c0 = arith.constant 0 : index
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 6 {
        affine.for %arg5 = 0 to 58 {
          affine.for %arg6 = 0 to 58 step 29 {
            %0 = ADORA.BlockLoad %arg2 [%arg3, %arg4, %arg5, %arg6] : memref<1x6x58x58xf32> -> memref<1x1x1x30xf32>  {Id = "0", KernelName = "conv2d_kernel_0"}
            %1 = ADORA.BlockLoad %arg0 [%arg3, 0, %arg5, %arg6] : memref<1x3x64x64xf32> -> memref<1x3x7x36xf32>  {Id = "1", KernelName = "conv2d_kernel_0"}
            %2 = ADORA.BlockLoad %arg1 [%arg4, 0, 0, 0] : memref<6x3x7x7xf32> -> memref<2x3x7x7xf32>  {Id = "2", KernelName = "conv2d_kernel_0"}
            %3 = ADORA.LocalMemAlloc memref<1x1x1x30xf32>  {Id = "3", KernelName = "conv2d_kernel_0"}
            ADORA.kernel {
              affine.for %arg7 = 0 to 29 {
                %4 = affine.load %0[0, 0, 0, %arg7] : memref<1x1x1x30xf32>
                %5 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %4) -> (f32) {
                  %6 = affine.for %arg10 = 0 to 7 iter_args(%arg11 = %arg9) -> (f32) {
                    %7 = affine.load %1[0, %arg8, %arg10, %arg7] : memref<1x3x7x36xf32>
                    %8 = affine.load %2[0, %arg8, %arg10, 0] : memref<2x3x7x7xf32>
                    %9 = arith.mulf %7, %8 : f32
                    %10 = arith.addf %arg11, %9 : f32
                    %c1 = arith.constant 1 : index
                    %11 = affine.load %1[0, %arg8, %arg10, %arg7 + 1] : memref<1x3x7x36xf32>
                    %12 = affine.load %2[0, %arg8, %arg10, 1] : memref<2x3x7x7xf32>
                    %13 = arith.mulf %11, %12 : f32
                    %14 = arith.addf %10, %13 : f32
                    %c2 = arith.constant 2 : index
                    %15 = affine.load %1[0, %arg8, %arg10, %arg7 + 2] : memref<1x3x7x36xf32>
                    %16 = affine.load %2[0, %arg8, %arg10, 2] : memref<2x3x7x7xf32>
                    %17 = arith.mulf %15, %16 : f32
                    %18 = arith.addf %14, %17 : f32
                    %c3 = arith.constant 3 : index
                    %19 = affine.load %1[0, %arg8, %arg10, %arg7 + 3] : memref<1x3x7x36xf32>
                    %20 = affine.load %2[0, %arg8, %arg10, 3] : memref<2x3x7x7xf32>
                    %21 = arith.mulf %19, %20 : f32
                    %22 = arith.addf %18, %21 : f32
                    %c4 = arith.constant 4 : index
                    %23 = affine.load %1[0, %arg8, %arg10, %arg7 + 4] : memref<1x3x7x36xf32>
                    %24 = affine.load %2[0, %arg8, %arg10, 4] : memref<2x3x7x7xf32>
                    %25 = arith.mulf %23, %24 : f32
                    %26 = arith.addf %22, %25 : f32
                    %c5 = arith.constant 5 : index
                    %27 = affine.load %1[0, %arg8, %arg10, %arg7 + 5] : memref<1x3x7x36xf32>
                    %28 = affine.load %2[0, %arg8, %arg10, 5] : memref<2x3x7x7xf32>
                    %29 = arith.mulf %27, %28 : f32
                    %30 = arith.addf %26, %29 : f32
                    %c6 = arith.constant 6 : index
                    %31 = affine.load %1[0, %arg8, %arg10, %arg7 + 6] : memref<1x3x7x36xf32>
                    %32 = affine.load %2[0, %arg8, %arg10, 6] : memref<2x3x7x7xf32>
                    %33 = arith.mulf %31, %32 : f32
                    %34 = arith.addf %30, %33 : f32
                    affine.yield %34 : f32
                  }
                  affine.yield %6 : f32
                }
                affine.store %5, %3[0, 0, 0, %arg7] : memref<1x1x1x30xf32>
              }
              ADORA.terminator
            } {KernelName = "conv2d_kernel_0"}
            ADORA.BlockStore %3, %arg2 [0, %arg4, %arg5, %arg6] : memref<1x1x1x30xf32> -> memref<1x6x58x58xf32>  {Id = "3", KernelName = "conv2d_kernel_0"}
          }
        }
      }
    }
    return
  }
}
