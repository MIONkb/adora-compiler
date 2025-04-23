module attributes {torch.debug_module_name = "Softmax"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<1x128x64xf32>, %arg1: memref<1x128x1xf32>, %arg2: memref<1x128x1xi64>, %arg3: memref<1x128x64xf32>, %arg4: memref<1x128x64xf32>, %arg5: memref<1x128x1xf32>, %arg6: memref<1x128x1xf32>, %arg7: memref<1x128x64xf32>) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 128 {
        affine.for %arg10 = 0 to 64 {
          %0 = affine.load %arg0[%arg8, %arg9, %arg10] : memref<1x128x64xf32>
          %1 = affine.load %arg1[%arg8, %arg9, 0] : memref<1x128x1xf32>
          %2 = affine.load %arg2[%arg8, %arg9, 0] : memref<1x128x1xi64>
          %3 = arith.index_cast %arg10 : index to i64
          %4 = arith.maximumf %0, %1 : f32
          %5 = arith.cmpf ogt, %0, %1 : f32
          %6 = arith.select %5, %3, %2 : i64
          affine.store %4, %arg1[%arg8, %arg9, 0] : memref<1x128x1xf32>
          affine.store %6, %arg2[%arg8, %arg9, 0] : memref<1x128x1xi64>
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 128 {
        affine.for %arg10 = 0 to 64 {
          %0 = affine.load %arg0[0, %arg9, %arg10] : memref<1x128x64xf32>
          %1 = affine.load %arg1[0, %arg9, 0] : memref<1x128x1xf32>
          %2 = arith.subf %0, %1 : f32
          affine.store %2, %arg3[%arg8, %arg9, %arg10] : memref<1x128x64xf32>
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 128 {
        affine.for %arg10 = 0 to 64 {
          %0 = affine.load %arg3[0, %arg9, %arg10] : memref<1x128x64xf32>
          %1 = math.rsqrt %0 : f32
          affine.store %1, %arg4[%arg8, %arg9, %arg10] : memref<1x128x64xf32>
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 128 {
        affine.for %arg10 = 0 to 64 {
          %0 = affine.load %arg4[%arg8, %arg9, %arg10] : memref<1x128x64xf32>
          %1 = affine.load %arg6[%arg8, %arg9, 0] : memref<1x128x1xf32>
          %2 = arith.addf %0, %1 : f32
          affine.store %2, %arg6[%arg8, %arg9, 0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 128 {
        affine.for %arg10 = 0 to 64 {
          %0 = affine.load %arg4[0, %arg9, %arg10] : memref<1x128x64xf32>
          %1 = affine.load %arg6[0, %arg9, 0] : memref<1x128x1xf32>
          %2 = arith.divf %0, %1 : f32
          affine.store %2, %arg7[%arg8, %arg9, %arg10] : memref<1x128x64xf32>
        }
      }
    }
    return
  }
}

