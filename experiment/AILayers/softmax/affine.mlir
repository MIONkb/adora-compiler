module attributes {torch.debug_module_name = "Softmax"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xi64>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 1 {
          affine.store %c0_i64, %alloc[%arg1, %arg2, %arg3] : memref<1x128x1xi64>
        }
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 1 {
          affine.store %cst, %alloc_1[%arg1, %arg2, %arg3] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    memref.copy %alloc_1, %alloc_2 : memref<1x128x1xf32> to memref<1x128x1xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xi64>
    memref.copy %alloc, %alloc_3 : memref<1x128x1xi64> to memref<1x128x1xi64>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 64 {
          %0 = affine.load %arg0[%arg1, %arg2, %arg3] : memref<1x128x64xf32>
          %1 = affine.load %alloc_2[%arg1, %arg2, 0] : memref<1x128x1xf32>
          %2 = affine.load %alloc_3[%arg1, %arg2, 0] : memref<1x128x1xi64>
          %3 = arith.index_cast %arg3 : index to i64
          %4 = arith.maximumf %0, %1 : f32
          %5 = arith.cmpf ogt, %0, %1 : f32
          %6 = arith.select %5, %3, %2 : i64
          affine.store %4, %alloc_2[%arg1, %arg2, 0] : memref<1x128x1xf32>
          affine.store %6, %alloc_3[%arg1, %arg2, 0] : memref<1x128x1xi64>
        }
      }
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 64 {
          %0 = affine.load %arg0[0, %arg2, %arg3] : memref<1x128x64xf32>
          %1 = affine.load %alloc_2[0, %arg2, 0] : memref<1x128x1xf32>
          %2 = arith.subf %0, %1 : f32
          affine.store %2, %alloc_4[%arg1, %arg2, %arg3] : memref<1x128x64xf32>
        }
      }
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 64 {
          %0 = affine.load %alloc_4[0, %arg2, %arg3] : memref<1x128x64xf32>
          %1 = math.exp %0 : f32
          affine.store %1, %alloc_5[%arg1, %arg2, %arg3] : memref<1x128x64xf32>
        }
      }
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 1 {
          affine.store %cst_0, %alloc_6[%arg1, %arg2, %arg3] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    memref.copy %alloc_6, %alloc_7 : memref<1x128x1xf32> to memref<1x128x1xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 64 {
          %0 = affine.load %alloc_5[%arg1, %arg2, %arg3] : memref<1x128x64xf32>
          %1 = affine.load %alloc_7[%arg1, %arg2, 0] : memref<1x128x1xf32>
          %2 = arith.addf %0, %1 : f32
          affine.store %2, %alloc_7[%arg1, %arg2, 0] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 64 {
          %0 = affine.load %alloc_5[0, %arg2, %arg3] : memref<1x128x64xf32>
          %1 = affine.load %alloc_7[0, %arg2, 0] : memref<1x128x1xf32>
          %2 = arith.divf %0, %1 : f32
          affine.store %2, %alloc_8[%arg1, %arg2, %arg3] : memref<1x128x64xf32>
        }
      }
    }
    return %alloc_8 : memref<1x128x64xf32>
  }
}

