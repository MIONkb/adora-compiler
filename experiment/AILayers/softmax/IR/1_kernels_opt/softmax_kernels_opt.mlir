module attributes {torch.debug_module_name = "Softmax"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<1x128x64xf32>, %arg1: memref<1x128x1xf32>, %arg2: memref<1x128x1xi64>, %arg3: memref<1x128x64xf32>, %arg4: memref<1x128x64xf32>, %arg5: memref<1x128x1xf32>, %arg6: memref<1x128x1xf32>, %arg7: memref<1x128x64xf32>) {
    affine.for %arg8 = 0 to 1 {
      %cst = arith.constant 1.500000e+00 : f32
      affine.for %arg9 = 0 to 128 step 32 {
        %0 = ADORA.BlockLoad %arg1 [%arg8, %arg9, 0] : memref<1x128x1xf32> -> memref<1x32x1xf32>  {Id = "0", KernelName = "forward_0"}
        %1 = ADORA.BlockLoad %arg0 [%arg8, %arg9, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "1", KernelName = "forward_0"}
        %2 = ADORA.LocalMemAlloc memref<1x32x1xf32>  {Id = "2", KernelName = "forward_0"}
        ADORA.kernel {
          affine.for %arg10 = 0 to 32 {
            // %3 = affine.load %0[0, %arg10, 0] : memref<1x32x1xf32>
            %4 = affine.for %arg11 = 0 to 64 iter_args(%arg12 = %cst) -> (f32) {
              %5 = affine.load %1[0, %arg10, %arg11] : memref<1x32x64xf32>
              %6 = arith.cmpf ugt, %5, %arg12 : f32
              %7 = arith.select %6, %5, %arg12 : f32
              %8 = arith.cmpf ugt, %arg12, %cst : f32
              %9 = arith.select %8, %arg12, %7 : f32
              affine.yield %9 : f32
            }
            affine.store %4, %2[%arg8, %arg10, 0] : memref<1x32x1xf32>
          }
          ADORA.terminator
        } {KernelName = "forward_0"}
        ADORA.BlockStore %2, %arg1 [0, %arg9, 0] : memref<1x32x1xf32> -> memref<1x128x1xf32>  {Id = "2", KernelName = "forward_0"}
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 128 step 32 {
        %0 = ADORA.BlockLoad %arg0 [0, %arg9, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "0", KernelName = "forward_1"}
        %1 = ADORA.BlockLoad %arg1 [0, %arg9, 0] : memref<1x128x1xf32> -> memref<1x32x1xf32>  {Id = "1", KernelName = "forward_1"}
        %2 = ADORA.LocalMemAlloc memref<1x32x64xf32>  {Id = "2", KernelName = "forward_1"}
        ADORA.kernel {
          affine.for %arg10 = 0 to 32 {
            affine.for %arg11 = 0 to 64 {
              %3 = affine.load %0[0, %arg10, %arg11] : memref<1x32x64xf32>
              %4 = affine.load %1[0, %arg10, 0] : memref<1x32x1xf32>
              %5 = arith.subf %3, %4 : f32
              affine.store %5, %2[0, %arg10, %arg11] : memref<1x32x64xf32>
            }
          }
          ADORA.terminator
        } {KernelName = "forward_1"}
        ADORA.BlockStore %2, %arg3 [0, %arg9, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "2", KernelName = "forward_1"}
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 128 step 32 {
        %0 = ADORA.BlockLoad %arg3 [0, %arg9, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "0", KernelName = "forward_2"}
        %1 = ADORA.LocalMemAlloc memref<1x32x64xf32>  {Id = "1", KernelName = "forward_2"}
        ADORA.kernel {
          affine.for %arg10 = 0 to 32 {
            affine.for %arg11 = 0 to 64 {
              %2 = affine.load %0[0, %arg10, %arg11] : memref<1x32x64xf32>
              %cst = arith.constant 5.000000e-01 : f32
              %3 = arith.mulf %2, %cst : f32
              %4 = arith.bitcast %2 : f32 to i32
              %c1_i32 = arith.constant 1 : i32
              %5 = arith.shrui %4, %c1_i32 : i32
              %c1597463007_i32 = arith.constant 1597463007 : i32
              %6 = arith.subi %c1597463007_i32, %5 : i32
              %7 = arith.bitcast %6 : i32 to f32
              %cst_0 = arith.constant 1.500000e+00 : f32
              %8 = arith.mulf %7, %7 : f32
              %9 = arith.mulf %8, %3 : f32
              %10 = arith.subf %cst_0, %9 : f32
              %11 = arith.mulf %10, %8 : f32
              affine.store %11, %1[0, %arg10, %arg11] : memref<1x32x64xf32>
            }
          }
          ADORA.terminator
        } {KernelName = "forward_2"}
        ADORA.BlockStore %1, %arg4 [0, %arg9, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "1", KernelName = "forward_2"}
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 128 step 32 {
        %0 = ADORA.BlockLoad %arg6 [%arg8, %arg9, 0] : memref<1x128x1xf32> -> memref<1x32x1xf32>  {Id = "0", KernelName = "forward_3"}
        %1 = ADORA.BlockLoad %arg4 [%arg8, %arg9, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "1", KernelName = "forward_3"}
        %2 = ADORA.LocalMemAlloc memref<1x32x1xf32>  {Id = "2", KernelName = "forward_3"}
        ADORA.kernel {
          affine.for %arg10 = 0 to 32 {
            %3 = affine.load %0[0, %arg10, 0] : memref<1x32x1xf32>
            %4 = affine.for %arg11 = 0 to 64 iter_args(%arg12 = %3) -> (f32) {
              %5 = affine.load %1[0, %arg10, %arg11] : memref<1x32x64xf32>
              %6 = arith.addf %5, %arg12 : f32
              affine.yield %6 : f32
            }
            affine.store %4, %2[%arg8, %arg10, 0] : memref<1x32x1xf32>
          }
          ADORA.terminator
        } {KernelName = "forward_3"}
        ADORA.BlockStore %2, %arg6 [0, %arg9, 0] : memref<1x32x1xf32> -> memref<1x128x1xf32>  {Id = "2", KernelName = "forward_3"}
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 128 step 32 {
        %0 = ADORA.BlockLoad %arg4 [0, %arg9, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "0", KernelName = "forward_4"}
        %1 = ADORA.BlockLoad %arg6 [0, %arg9, 0] : memref<1x128x1xf32> -> memref<1x32x1xf32>  {Id = "1", KernelName = "forward_4"}
        %2 = ADORA.LocalMemAlloc memref<1x32x64xf32>  {Id = "2", KernelName = "forward_4"}
        ADORA.kernel {
          affine.for %arg10 = 0 to 32 {
            affine.for %arg11 = 0 to 64 {
              %3 = affine.load %0[0, %arg10, %arg11] : memref<1x32x64xf32>
              %4 = affine.load %1[0, %arg10, 0] : memref<1x32x1xf32>
              %5 = arith.divf %3, %4 : f32
              affine.store %5, %2[0, %arg10, %arg11] : memref<1x32x64xf32>
            }
          }
          ADORA.terminator
        } {KernelName = "forward_4"}
        ADORA.BlockStore %2, %arg7 [0, %arg9, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "2", KernelName = "forward_4"}
      }
    }
    return
  }
}

