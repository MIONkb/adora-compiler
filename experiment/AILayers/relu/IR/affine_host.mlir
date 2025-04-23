module attributes {torch.debug_module_name = "ReLU"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %alloc) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %alloc : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}

