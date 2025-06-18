func.func @conv2d_kernel_0(%arg0: memref<1x3x64x64xf32>, %arg1: memref<6x3x7x7xf32>, %arg2: memref<1x6x58x58xf32>) attributes {Kernel, conv2d_kernel_0} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  affine.for %arg3 = 0 to 1 {
    affine.for %arg4 = 0 to 6 {
      affine.for %arg5 = 0 to 58 {

        affine.for %arg6 = 0 to 58 {
          affine.for %arg7 = 0 to 3 {
            affine.for %arg8 = 0 to 7 {
              affine.for %arg9 = 0 to 7 {
                %0 = affine.load %arg0[%arg3, %arg7, %arg5 + %arg8, %arg6 + %arg9] : memref<1x3x64x64xf32>
                %1 = affine.load %arg1[%arg4, %arg7, %arg8, %arg9] : memref<6x3x7x7xf32>
                %2 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<1x6x58x57xf32>
                %3 = arith.mulf %0, %1 : f32
                %4 = arith.addf %2, %3 : f32
                affine.store %4, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<1x6x58x58xf32>
              }
            }
          }
        }

      }
    }
  }
  return
}