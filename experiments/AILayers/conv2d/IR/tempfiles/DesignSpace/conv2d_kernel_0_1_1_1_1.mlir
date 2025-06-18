module {
  func.func @conv2d_kernel_0(%arg0: memref<1x3x64x64xf32>, %arg1: memref<6x3x7x7xf32>, %arg2: memref<1x6x58x58xf32>) attributes {Kernel, conv2d_kernel_0} {
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
                    %7 = affine.for %arg12 = 0 to 7 iter_args(%arg13 = %arg11) -> (f32) {
                      %8 = affine.load %1[0, %arg8, %arg10, %arg7 + %arg12] : memref<1x3x7x36xf32>
                      %9 = affine.load %2[0, %arg8, %arg10, %arg12] : memref<2x3x7x7xf32>
                      %10 = arith.mulf %8, %9 : f32
                      %11 = arith.addf %arg13, %10 : f32
                      affine.yield %11 : f32
                    }
                    affine.yield %7 : f32
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
