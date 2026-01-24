#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_0(%arg0: memref<8x3072xbf16>, %arg1: memref<3072x768xbf16>, %arg2: memref<8x768xbf16>) -> memref<8x768xbf16> {
  %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {linalg.memoized_indexing_maps = [#map3, #map4, #map5], operandSegmentSizes = array<i32: 2, 1>, stationary_kind = "InputStationary", tile_size = array<i64: 2 ,128, 4, 4>} : (memref<8x3072xbf16>, memref<3072x768xbf16>, memref<8x768xbf16>) -> memref<8x768xbf16>
  return %0 : memref<8x768xbf16>
}