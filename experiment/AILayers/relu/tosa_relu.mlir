module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: tensor<1x128x64xf32>) -> tensor<1x128x64xf32> {
    %0 = tosa.clamp %arg0 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
    return %0 : tensor<1x128x64xf32>
  }
}

