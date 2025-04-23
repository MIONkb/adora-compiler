module attributes {torch.debug_module_name = "Softmax"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<1x128x64xf32>, %arg1: memref<1x128x1xf32>, %arg2: memref<1x128x1xi64>, %arg3: memref<1x128x64xf32>, %arg4: memref<1x128x64xf32>, %arg5: memref<1x128x1xf32>, %arg6: memref<1x128x1xf32>, %arg7: memref<1x128x64xf32>) {
    ADORA.kernel {
      affine.for %arg8 = 0 to 1 {
        %cst_0 = arith.constant 1.500000e+00 : f32
        affine.for %arg9 = 0 to 128 {
          affine.for %arg10 = 0 to 64 {
            %0 = affine.load %arg0[%arg8, %arg9, %arg10] : memref<1x128x64xf32>
            %1 = affine.load %arg1[%arg8, %arg9, 0] : memref<1x128x1xf32>
            %2 = arith.cmpf ugt, %0, %1 : f32
            %3 = arith.select %2, %0, %1 : f32
            // %4 = arith.cmpf uno, %1, %1 : f32
            %4 = arith.cmpf ugt, %1, %cst_0 : f32
            %5 = arith.select %4, %1, %3 : f32
            affine.store %5, %arg1[%arg8, %arg9, 0] : memref<1x128x1xf32>
          }
        }
      }
      ADORA.terminator
    } {KernelName = "forward_0"}
    ADORA.kernel {
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
      ADORA.terminator
    } {KernelName = "forward_1"}
    ADORA.kernel {
      affine.for %arg8 = 0 to 1 {
        affine.for %arg9 = 0 to 128 {
          affine.for %arg10 = 0 to 64 {
            %0 = affine.load %arg3[0, %arg9, %arg10] : memref<1x128x64xf32>
            %1 = math.rsqrt %0 : f32
            affine.store %1, %arg4[%arg8, %arg9, %arg10] : memref<1x128x64xf32>
          }
        }
      }
      ADORA.terminator
    } {KernelName = "forward_2"}
    ADORA.kernel {
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
      ADORA.terminator
    } {KernelName = "forward_3"}
    ADORA.kernel {
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
      ADORA.terminator
    } {KernelName = "forward_4"}
    return
  }
}

