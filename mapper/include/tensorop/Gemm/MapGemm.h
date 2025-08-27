#ifndef ADORA_TENSOR_GEMM_OP_MAP_H
#define ADORA_TENSOR_GEMM_OP_MAP_H
#include "mlir/Dialect/Affine/IR/AffineOps.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"

namespace mlir{
namespace ADORA{

/////
//// WS Order : N (j) -> K (k) -> M (i)
////  M - row of A, row of C
////  N - col of B, col of C
////  K - reduction dim
affine::AffineForOp TiledWeightStationaryGemm(
  ADORATensor::GemmOp op, 
  ArrayRef<int64_t> tilesize
);

}
}

#endif // ADORA_TENSOR_GEMM_OP_MAP_H