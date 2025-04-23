module attributes {torch.debug_module_name = "Softmax"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<1x128x64xf32>, %arg1: memref<1x128x1xf32>, %arg2: memref<1x128x1xi64>, %arg3: memref<1x128x64xf32>, %arg4: memref<1x128x64xf32>, %arg5: memref<1x128x1xf32>, %arg6: memref<1x128x1xf32>, %arg7: memref<1x128x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    call @forward_kernel_0(%arg0, %arg1, %arg2) : (memref<1x128x64xf32>, memref<1x128x1xf32>, memref<1x128x1xi64>) -> ()
    call @forward_kernel_1(%arg0, %arg1, %arg3) : (memref<1x128x64xf32>, memref<1x128x1xf32>, memref<1x128x64xf32>) -> ()
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 128 {
        affine.for %arg10 = 0 to 64 {
          %0 = affine.load %arg3[0, %arg9, %arg10] : memref<1x128x64xf32>
          %1 = math.exp %0 : f32
          affine.store %1, %arg4[%arg8, %arg9, %arg10] : memref<1x128x64xf32>
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 128 {
        affine.for %arg10 = 0 to 1 {
          affine.store %cst, %arg5[%arg8, %arg9, %arg10] : memref<1x128x1xf32>
        }
      }
    }
    call @forward_kernel_2(%arg4, %arg6) : (memref<1x128x64xf32>, memref<1x128x1xf32>) -> ()
    call @forward_kernel_3(%arg4, %arg6, %arg7) : (memref<1x128x64xf32>, memref<1x128x1xf32>, memref<1x128x64xf32>) -> ()
    return
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x1xf32>, memref<1x128x1xi64>)
  func.func private @forward_kernel_1(memref<1x128x64xf32>, memref<1x128x1xf32>, memref<1x128x64xf32>)
  func.func private @forward_kernel_2(memref<1x128x64xf32>, memref<1x128x1xf32>)
  func.func private @forward_kernel_3(memref<1x128x64xf32>, memref<1x128x1xf32>, memref<1x128x64xf32>)
}

