module attributes {torch.debug_module_name = "ReLU"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 64 {
          %0 = affine.load %arg0[0, %arg2, %arg3] : memref<1x128x64xf32>
          %1 = arith.cmpf ugt, %0, %cst : f32
          %2 = arith.select %1, %0, %cst : f32
          affine.store %2, %alloc[%arg1, %arg2, %arg3] : memref<1x128x64xf32>
        }
      }
    }
    return %alloc : memref<1x128x64xf32>
  }
}

