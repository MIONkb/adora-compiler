// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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


// -----// IR Dump After EmptyTensorElimination (eliminate-empty-tensors) //----- //
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


// -----// IR Dump After EmptyTensorToAllocTensor (empty-tensor-to-alloc-tensor) //----- //
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
    %0 = bufferization.alloc_tensor() : tensor<1x128x1xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1x128x1xi64>) -> tensor<1x128x1xi64>
    %2 = bufferization.alloc_tensor() : tensor<1x128x1xf32>
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
    %5 = bufferization.alloc_tensor() : tensor<1x128x64xf32>
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


// -----// IR Dump After ArithBufferize (arith-bufferize) //----- //
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
    %alloc = memref.alloc() : memref<1x128x1xi64>
    %0 = bufferization.to_tensor %alloc : memref<1x128x1xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1x128x1xi64>) -> tensor<1x128x1xi64>
    %alloc_1 = memref.alloc() : memref<1x128x1xf32>
    %2 = bufferization.to_tensor %alloc_1 : memref<1x128x1xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
    %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x128x64xf32>) outs(%3, %1 : tensor<1x128x1xf32>, tensor<1x128x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_3: i64):
      %11 = linalg.index 2 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = arith.maximumf %in, %out : f32
      %14 = arith.cmpf ogt, %in, %out : f32
      %15 = arith.select %14, %12, %out_3 : i64
      linalg.yield %13, %15 : f32, i64
    } -> (tensor<1x128x1xf32>, tensor<1x128x1xi64>)
    %alloc_2 = memref.alloc() : memref<1x128x64xf32>
    %5 = bufferization.to_tensor %alloc_2 : memref<1x128x64xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %4#0 : tensor<1x128x64xf32>, tensor<1x128x1xf32>) outs(%5 : tensor<1x128x64xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %11 = arith.subf %in, %in_3 : f32
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
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %11 = arith.divf %in, %in_3 : f32
      linalg.yield %11 : f32
    } -> tensor<1x128x64xf32>
    return %10 : tensor<1x128x64xf32>
  }
}


// -----// IR Dump After TensorBufferize (tensor-bufferize) //----- //
func.func @forward(%arg0: tensor<1x128x64xf32>) -> tensor<1x128x64xf32> {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<1x128x1xi64>
  %0 = bufferization.to_tensor %alloc : memref<1x128x1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1x128x1xi64>) -> tensor<1x128x1xi64>
  %alloc_1 = memref.alloc() : memref<1x128x1xf32>
  %2 = bufferization.to_tensor %alloc_1 : memref<1x128x1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x128x64xf32>) outs(%3, %1 : tensor<1x128x1xf32>, tensor<1x128x1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_3: i64):
    %11 = linalg.index 2 : index
    %12 = arith.index_cast %11 : index to i64
    %13 = arith.maximumf %in, %out : f32
    %14 = arith.cmpf ogt, %in, %out : f32
    %15 = arith.select %14, %12, %out_3 : i64
    linalg.yield %13, %15 : f32, i64
  } -> (tensor<1x128x1xf32>, tensor<1x128x1xi64>)
  %alloc_2 = memref.alloc() : memref<1x128x64xf32>
  %5 = bufferization.to_tensor %alloc_2 : memref<1x128x64xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %4#0 : tensor<1x128x64xf32>, tensor<1x128x1xf32>) outs(%5 : tensor<1x128x64xf32>) {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %11 = arith.subf %in, %in_3 : f32
    linalg.yield %11 : f32
  } -> tensor<1x128x64xf32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6 : tensor<1x128x64xf32>) outs(%5 : tensor<1x128x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %11 = math.exp %in : f32
    linalg.yield %11 : f32
  } -> tensor<1x128x64xf32>
  %8 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7 : tensor<1x128x64xf32>) outs(%8 : tensor<1x128x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %11 = arith.addf %in, %out : f32
    linalg.yield %11 : f32
  } -> tensor<1x128x1xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %9 : tensor<1x128x64xf32>, tensor<1x128x1xf32>) outs(%5 : tensor<1x128x64xf32>) {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %11 = arith.divf %in, %in_3 : f32
    linalg.yield %11 : f32
  } -> tensor<1x128x64xf32>
  return %10 : tensor<1x128x64xf32>
}

// -----// IR Dump After FuncBufferize (func-bufferize) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
module attributes {torch.debug_module_name = "Softmax"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %0 = bufferization.to_tensor %arg0 : memref<1x128x64xf32>
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<1x128x1xi64>
    %1 = bufferization.to_tensor %alloc : memref<1x128x1xi64>
    %2 = linalg.fill ins(%c0_i64 : i64) outs(%1 : tensor<1x128x1xi64>) -> tensor<1x128x1xi64>
    %alloc_1 = memref.alloc() : memref<1x128x1xf32>
    %3 = bufferization.to_tensor %alloc_1 : memref<1x128x1xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
    %5:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0 : tensor<1x128x64xf32>) outs(%4, %2 : tensor<1x128x1xf32>, tensor<1x128x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_3: i64):
      %13 = linalg.index 2 : index
      %14 = arith.index_cast %13 : index to i64
      %15 = arith.maximumf %in, %out : f32
      %16 = arith.cmpf ogt, %in, %out : f32
      %17 = arith.select %16, %14, %out_3 : i64
      linalg.yield %15, %17 : f32, i64
    } -> (tensor<1x128x1xf32>, tensor<1x128x1xi64>)
    %alloc_2 = memref.alloc() : memref<1x128x64xf32>
    %6 = bufferization.to_tensor %alloc_2 : memref<1x128x64xf32>
    %7 = linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %5#0 : tensor<1x128x64xf32>, tensor<1x128x1xf32>) outs(%6 : tensor<1x128x64xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %13 = arith.subf %in, %in_3 : f32
      linalg.yield %13 : f32
    } -> tensor<1x128x64xf32>
    %8 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<1x128x64xf32>) outs(%6 : tensor<1x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %13 = math.exp %in : f32
      linalg.yield %13 : f32
    } -> tensor<1x128x64xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%8 : tensor<1x128x64xf32>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %13 = arith.addf %in, %out : f32
      linalg.yield %13 : f32
    } -> tensor<1x128x1xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %10 : tensor<1x128x64xf32>, tensor<1x128x1xf32>) outs(%6 : tensor<1x128x64xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %13 = arith.divf %in, %in_3 : f32
      linalg.yield %13 : f32
    } -> tensor<1x128x64xf32>
    %12 = bufferization.to_memref %11 : memref<1x128x64xf32>
    return %12 : memref<1x128x64xf32>
  }
}


// -----// IR Dump After LinalgBufferize (linalg-bufferize) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
module attributes {torch.debug_module_name = "Softmax"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<1x128x1xi64>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xi64>
    linalg.fill ins(%c0_i64 : i64) outs(%alloc_1 : memref<1x128x1xi64>)
    %alloc_2 = memref.alloc() : memref<1x128x1xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<1x128x1xf32>)
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    memref.copy %alloc_3, %alloc_4 : memref<1x128x1xf32> to memref<1x128x1xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xi64>
    memref.copy %alloc_1, %alloc_5 : memref<1x128x1xi64> to memref<1x128x1xi64>
    linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : memref<1x128x64xf32>) outs(%alloc_4, %alloc_5 : memref<1x128x1xf32>, memref<1x128x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_12: i64):
      %0 = linalg.index 2 : index
      %1 = arith.index_cast %0 : index to i64
      %2 = arith.maximumf %in, %out : f32
      %3 = arith.cmpf ogt, %in, %out : f32
      %4 = arith.select %3, %1, %out_12 : i64
      linalg.yield %2, %4 : f32, i64
    }
    %alloc_6 = memref.alloc() : memref<1x128x64xf32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %alloc_4 : memref<1x128x64xf32>, memref<1x128x1xf32>) outs(%alloc_7 : memref<1x128x64xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %0 = arith.subf %in, %in_12 : f32
      linalg.yield %0 : f32
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_7 : memref<1x128x64xf32>) outs(%alloc_8 : memref<1x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = math.exp %in : f32
      linalg.yield %0 : f32
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.fill ins(%cst_0 : f32) outs(%alloc_9 : memref<1x128x1xf32>)
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    memref.copy %alloc_9, %alloc_10 : memref<1x128x1xf32> to memref<1x128x1xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%alloc_8 : memref<1x128x64xf32>) outs(%alloc_10 : memref<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    }
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_8, %alloc_10 : memref<1x128x64xf32>, memref<1x128x1xf32>) outs(%alloc_11 : memref<1x128x64xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %0 = arith.divf %in, %in_12 : f32
      linalg.yield %0 : f32
    }
    return %alloc_11 : memref<1x128x64xf32>
  }
}


// -----// IR Dump After FinalizingBufferize (finalizing-bufferize) //----- //
func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<1x128x1xi64>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xi64>
  linalg.fill ins(%c0_i64 : i64) outs(%alloc_1 : memref<1x128x1xi64>)
  %alloc_2 = memref.alloc() : memref<1x128x1xf32>
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
  linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<1x128x1xf32>)
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
  memref.copy %alloc_3, %alloc_4 : memref<1x128x1xf32> to memref<1x128x1xf32>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xi64>
  memref.copy %alloc_1, %alloc_5 : memref<1x128x1xi64> to memref<1x128x1xi64>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : memref<1x128x64xf32>) outs(%alloc_4, %alloc_5 : memref<1x128x1xf32>, memref<1x128x1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_12: i64):
    %0 = linalg.index 2 : index
    %1 = arith.index_cast %0 : index to i64
    %2 = arith.maximumf %in, %out : f32
    %3 = arith.cmpf ogt, %in, %out : f32
    %4 = arith.select %3, %1, %out_12 : i64
    linalg.yield %2, %4 : f32, i64
  }
  %alloc_6 = memref.alloc() : memref<1x128x64xf32>
  %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %alloc_4 : memref<1x128x64xf32>, memref<1x128x1xf32>) outs(%alloc_7 : memref<1x128x64xf32>) {
  ^bb0(%in: f32, %in_12: f32, %out: f32):
    %0 = arith.subf %in, %in_12 : f32
    linalg.yield %0 : f32
  }
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_7 : memref<1x128x64xf32>) outs(%alloc_8 : memref<1x128x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = math.exp %in : f32
    linalg.yield %0 : f32
  }
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
  linalg.fill ins(%cst_0 : f32) outs(%alloc_9 : memref<1x128x1xf32>)
  %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
  memref.copy %alloc_9, %alloc_10 : memref<1x128x1xf32> to memref<1x128x1xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%alloc_8 : memref<1x128x64xf32>) outs(%alloc_10 : memref<1x128x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.addf %in, %out : f32
    linalg.yield %0 : f32
  }
  %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_8, %alloc_10 : memref<1x128x64xf32>, memref<1x128x1xf32>) outs(%alloc_11 : memref<1x128x64xf32>) {
  ^bb0(%in: f32, %in_12: f32, %out: f32):
    %0 = arith.divf %in, %in_12 : f32
    linalg.yield %0 : f32
  }
  return %alloc_11 : memref<1x128x64xf32>
}

// -----// IR Dump After ConvertBufferizationToMemRef (convert-bufferization-to-memref) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
module attributes {torch.debug_module_name = "Softmax"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<1x128x1xi64>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xi64>
    linalg.fill ins(%c0_i64 : i64) outs(%alloc_1 : memref<1x128x1xi64>)
    %alloc_2 = memref.alloc() : memref<1x128x1xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<1x128x1xf32>)
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    memref.copy %alloc_3, %alloc_4 : memref<1x128x1xf32> to memref<1x128x1xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xi64>
    memref.copy %alloc_1, %alloc_5 : memref<1x128x1xi64> to memref<1x128x1xi64>
    linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : memref<1x128x64xf32>) outs(%alloc_4, %alloc_5 : memref<1x128x1xf32>, memref<1x128x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_12: i64):
      %0 = linalg.index 2 : index
      %1 = arith.index_cast %0 : index to i64
      %2 = arith.maximumf %in, %out : f32
      %3 = arith.cmpf ogt, %in, %out : f32
      %4 = arith.select %3, %1, %out_12 : i64
      linalg.yield %2, %4 : f32, i64
    }
    %alloc_6 = memref.alloc() : memref<1x128x64xf32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %alloc_4 : memref<1x128x64xf32>, memref<1x128x1xf32>) outs(%alloc_7 : memref<1x128x64xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %0 = arith.subf %in, %in_12 : f32
      linalg.yield %0 : f32
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_7 : memref<1x128x64xf32>) outs(%alloc_8 : memref<1x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = math.exp %in : f32
      linalg.yield %0 : f32
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.fill ins(%cst_0 : f32) outs(%alloc_9 : memref<1x128x1xf32>)
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    memref.copy %alloc_9, %alloc_10 : memref<1x128x1xf32> to memref<1x128x1xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%alloc_8 : memref<1x128x64xf32>) outs(%alloc_10 : memref<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    }
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_8, %alloc_10 : memref<1x128x64xf32>, memref<1x128x1xf32>) outs(%alloc_11 : memref<1x128x64xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %0 = arith.divf %in, %in_12 : f32
      linalg.yield %0 : f32
    }
    return %alloc_11 : memref<1x128x64xf32>
  }
}


// -----// IR Dump After LinalgLowerToAffineLoops (convert-linalg-to-affine-loops) //----- //
module attributes {torch.debug_module_name = "Softmax"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %c0 = arith.constant 0 : index
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
          %1 = affine.load %alloc_2[%arg1, %arg2, %c0] : memref<1x128x1xf32>
          %2 = affine.load %alloc_3[%arg1, %arg2, %c0] : memref<1x128x1xi64>
          %3 = arith.index_cast %arg3 : index to i64
          %4 = arith.maximumf %0, %1 : f32
          %5 = arith.cmpf ogt, %0, %1 : f32
          %6 = arith.select %5, %3, %2 : i64
          affine.store %4, %alloc_2[%arg1, %arg2, %c0] : memref<1x128x1xf32>
          affine.store %6, %alloc_3[%arg1, %arg2, %c0] : memref<1x128x1xi64>
        }
      }
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 64 {
          %0 = affine.load %arg0[%c0, %arg2, %arg3] : memref<1x128x64xf32>
          %1 = affine.load %alloc_2[%c0, %arg2, %c0] : memref<1x128x1xf32>
          %2 = arith.subf %0, %1 : f32
          affine.store %2, %alloc_4[%arg1, %arg2, %arg3] : memref<1x128x64xf32>
        }
      }
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 64 {
          %0 = affine.load %alloc_4[%c0, %arg2, %arg3] : memref<1x128x64xf32>
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
          %1 = affine.load %alloc_7[%arg1, %arg2, %c0] : memref<1x128x1xf32>
          %2 = arith.addf %0, %1 : f32
          affine.store %2, %alloc_7[%arg1, %arg2, %c0] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 64 {
          %0 = affine.load %alloc_5[%c0, %arg2, %arg3] : memref<1x128x64xf32>
          %1 = affine.load %alloc_7[%c0, %arg2, %c0] : memref<1x128x1xf32>
          %2 = arith.divf %0, %1 : f32
          affine.store %2, %alloc_8[%arg1, %arg2, %arg3] : memref<1x128x64xf32>
        }
      }
    }
    return %alloc_8 : memref<1x128x64xf32>
  }
}


// -----// IR Dump After AffineScalarReplacement (affine-scalrep) //----- //
func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
  %c0 = arith.constant 0 : index
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
        %1 = affine.load %alloc_2[%arg1, %arg2, %c0] : memref<1x128x1xf32>
        %2 = affine.load %alloc_3[%arg1, %arg2, %c0] : memref<1x128x1xi64>
        %3 = arith.index_cast %arg3 : index to i64
        %4 = arith.maximumf %0, %1 : f32
        %5 = arith.cmpf ogt, %0, %1 : f32
        %6 = arith.select %5, %3, %2 : i64
        affine.store %4, %alloc_2[%arg1, %arg2, %c0] : memref<1x128x1xf32>
        affine.store %6, %alloc_3[%arg1, %arg2, %c0] : memref<1x128x1xi64>
      }
    }
  }
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 64 {
        %0 = affine.load %arg0[%c0, %arg2, %arg3] : memref<1x128x64xf32>
        %1 = affine.load %alloc_2[%c0, %arg2, %c0] : memref<1x128x1xf32>
        %2 = arith.subf %0, %1 : f32
        affine.store %2, %alloc_4[%arg1, %arg2, %arg3] : memref<1x128x64xf32>
      }
    }
  }
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 64 {
        %0 = affine.load %alloc_4[%c0, %arg2, %arg3] : memref<1x128x64xf32>
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
        %1 = affine.load %alloc_7[%arg1, %arg2, %c0] : memref<1x128x1xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc_7[%arg1, %arg2, %c0] : memref<1x128x1xf32>
      }
    }
  }
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 64 {
        %0 = affine.load %alloc_5[%c0, %arg2, %arg3] : memref<1x128x64xf32>
        %1 = affine.load %alloc_7[%c0, %arg2, %c0] : memref<1x128x1xf32>
        %2 = arith.divf %0, %1 : f32
        affine.store %2, %alloc_8[%arg1, %arg2, %arg3] : memref<1x128x64xf32>
      }
    }
  }
  return %alloc_8 : memref<1x128x64xf32>
}

// -----// IR Dump After SimplifyAffineStructures (affine-simplify-structures) //----- //
func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
  %c0 = arith.constant 0 : index
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
        %1 = affine.load %alloc_2[%arg1, %arg2, %c0] : memref<1x128x1xf32>
        %2 = affine.load %alloc_3[%arg1, %arg2, %c0] : memref<1x128x1xi64>
        %3 = arith.index_cast %arg3 : index to i64
        %4 = arith.maximumf %0, %1 : f32
        %5 = arith.cmpf ogt, %0, %1 : f32
        %6 = arith.select %5, %3, %2 : i64
        affine.store %4, %alloc_2[%arg1, %arg2, %c0] : memref<1x128x1xf32>
        affine.store %6, %alloc_3[%arg1, %arg2, %c0] : memref<1x128x1xi64>
      }
    }
  }
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 64 {
        %0 = affine.load %arg0[%c0, %arg2, %arg3] : memref<1x128x64xf32>
        %1 = affine.load %alloc_2[%c0, %arg2, %c0] : memref<1x128x1xf32>
        %2 = arith.subf %0, %1 : f32
        affine.store %2, %alloc_4[%arg1, %arg2, %arg3] : memref<1x128x64xf32>
      }
    }
  }
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 64 {
        %0 = affine.load %alloc_4[%c0, %arg2, %arg3] : memref<1x128x64xf32>
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
        %1 = affine.load %alloc_7[%arg1, %arg2, %c0] : memref<1x128x1xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc_7[%arg1, %arg2, %c0] : memref<1x128x1xf32>
      }
    }
  }
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 64 {
        %0 = affine.load %alloc_5[%c0, %arg2, %arg3] : memref<1x128x64xf32>
        %1 = affine.load %alloc_7[%c0, %arg2, %c0] : memref<1x128x1xf32>
        %2 = arith.divf %0, %1 : f32
        affine.store %2, %alloc_8[%arg1, %arg2, %arg3] : memref<1x128x64xf32>
      }
    }
  }
  return %alloc_8 : memref<1x128x64xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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


