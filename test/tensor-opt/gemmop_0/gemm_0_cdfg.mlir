// RUN: rm -f gemm_opt_0_CDFG.dot gemm_opt_1_CDFG.dot
// RUN: tensor-opt --adora-gen-tensor-op-cdfg %s

// RUN: test -s gemm_opt_1_CDFG.dot
// RUN: %FileCheck %s --check-prefix=DOT0 --input-file=gemm_opt_0_CDFG.dot

// DOT0: Digraph G
// DOT0-DAG: Input{{[0-9]+}} -> FMUL32{{[0-9]+}}
// DOT0-DAG: CONST{{[0-9]+}} -> FMUL32{{[0-9]+}}
// DOT0-DAG: FMUL32{{[0-9]+}} -> Output{{[0-9]+}}
// DOT0: }

func.func @matmul_0(%arg0: memref<8x3072xbf16>, %arg1: memref<3072x768xbf16>, %arg2: memref<8x768xbf16>) -> memref<8x768xbf16> {
   %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {stationary_kind = "InputStationary", tile_size = array<i64: 2, 768, 4, 12>} : (memref<8x3072xbf16>, memref<3072x768xbf16>, memref<8x768xbf16>) -> memref<8x768xbf16>
  return %0 : memref<8x768xbf16>
}