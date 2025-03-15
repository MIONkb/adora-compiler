#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
module attributes {torch.debug_module_name = "Softmax"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x128x64xf32>) -> tensor<1x128x64xf32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x128x1xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1x128x1xi64>) -> tensor<1x128x1xi64>
    %2 = tensor.empty() : tensor<1x128x1xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
    %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x128x64xf32>) outs(%3, %1 : tensor<1x128x1xf32>, tensor<1x128x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1: i64):
      %11 = linalg.index 2 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = arith.maximumf %in, %out : f32
      %14 = arith.cmpf ogt, %in, %out : f32
      %15 = arith.select %14, %12, %out_1 : i64
      linalg.yield %13, %15 : f32, i64
    } -> (tensor<1x128x1xf32>, tensor<1x128x1xi64>)
    %5 = tensor.empty() : tensor<1x128x64xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %4#0 : tensor<1x128x64xf32>, tensor<1x128x1xf32>) outs(%5 : tensor<1x128x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %11 = arith.subf %in, %in_1 : f32
      linalg.yield %11 : f32
    } -> tensor<1x128x64xf32>
    %7 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6 : tensor<1x128x64xf32>) outs(%5 : tensor<1x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = math.exp %in : f32
      linalg.yield %11 : f32
    } -> tensor<1x128x64xf32>
    %8 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7 : tensor<1x128x64xf32>) outs(%8 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = arith.addf %in, %out : f32
      linalg.yield %11 : f32
    } -> tensor<1x128x1xf32>
    %10 = linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %9 : tensor<1x128x64xf32>, tensor<1x128x1xf32>) outs(%5 : tensor<1x128x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %11 = arith.divf %in, %in_1 : f32
      linalg.yield %11 : f32
    } -> tensor<1x128x64xf32>
    return %10 : tensor<1x128x64xf32>
  }
}

