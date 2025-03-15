module attributes {torch.debug_module_name = "Softmax"} {
  func.func @forward(%arg0: tensor<1x128x64xf32>) -> tensor<1x128x64xf32> {
    %0 = tosa.reduce_max %arg0 {axis = 2 : i32} : (tensor<1x128x64xf32>) -> tensor<1x128x1xf32>
    %1 = tosa.sub %arg0, %0 : (tensor<1x128x64xf32>, tensor<1x128x1xf32>) -> tensor<1x128x64xf32>
    %2 = tosa.exp %1 : (tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
    %3 = tosa.reduce_sum %2 {axis = 2 : i32} : (tensor<1x128x64xf32>) -> tensor<1x128x1xf32>
    %4 = tosa.reciprocal %3 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
    %5 = tosa.mul %2, %4 {shift = 0 : i8} : (tensor<1x128x64xf32>, tensor<1x128x1xf32>) -> tensor<1x128x64xf32>
    return %5 : tensor<1x128x64xf32>
  }
}

