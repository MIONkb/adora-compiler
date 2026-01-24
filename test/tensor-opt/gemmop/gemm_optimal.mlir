#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul_0(%arg0: memref<8x3072xbf16>, %arg1: memref<3072x768xbf16>, %arg2: memref<8x768xbf16>) -> memref<8x768xbf16> {
    %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {linalg.memoized_indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>, stationary_kind = "OutputStationary", tile_size = array<i64: 1, 1536, 8, 4>} : (memref<8x3072xbf16>, memref<3072x768xbf16>, memref<8x768xbf16>) -> memref<8x768xbf16>
    return %0 : memref<8x768xbf16>
  }
}

