#ifndef ADORA_TENSOR_GEMM_OP_LOWER_H
#define ADORA_TENSOR_GEMM_OP_LOWER_H
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"

using StationaryBodyBuilderFn = std::function<void(mlir::OpBuilder &, mlir::Location, mlir::ValueRange)>;

namespace mlir{
namespace ADORA{

/// @brief Generate a nested affine.for loop on device(data transfer is already done).
///  This function is shared by IS and WS
affine::AffineForOp GenerateOnDeviceNestedLoop(
    OpBuilder &builder, Location loc,
    int level = 2, SmallVector<int> Upperbounds = {1, 1}, 
    StationaryBodyBuilderFn BodyBuilder = nullptr);

/// @brief Generate nested loop which is the outer loops out of a systolic tile
///  Shared by three stationary dataflow.
/// @param A            MemRef value representing the activation/input tensor.
/// @param B            MemRef value representing the weight tensor (stationary).
/// @param C            MemRef value representing the output/accumulation tensor.
/// @param tile_row_size Number of rows in the tile (micro-kernel row dimension).
/// @param tile_col_size Number of columns in the tile (micro-kernel column dimension).
affine::AffineForOp OffDeviceNestedLoop(
    OpBuilder &builder, Location loc,
    int level = 1, 
    SmallVector<int> Upperbounds = {1}, 
    SmallVector<int> Steps = {1}, 
    StationaryBodyBuilderFn InnerMostBodyBuilder = nullptr);

/////
//// WS Order : N (j) -> K (k) -> M (i)
////  M - row of A, row of C
////  N - col of B, col of C
////  K - reduction dim
mlir::affine::AffineForOp TiledWeightStationaryGemm(
  OpBuilder opbuilder, ADORATensor::GemmOp op, ArrayRef<int64_t> tilesize
);

/////
//// IS Order : K (k) -> M (i) -> N (j) 
////  K - reduction dim
////  M - row of A, row of C
////  N - col of B, col of C
//// when tile size = 4, (M_temporal_tile, N_temporal_tile, M_spatial_tile, K_spatial_tile)
//// when tile size = 3, (N_temporal_tile, M_spatial_tile, K_spatial_tile), M_temporal_tile == 1
//// when tile size = 2, (M_spatial_tile, K_spatial_tile), M_temporal_tile == 1, N_temporal_tile = N
mlir::affine::AffineForOp TiledInputStationaryGemm(
  OpBuilder opbuilder, ADORATensor::GemmOp op, ArrayRef<int64_t> tilesize //(K_temporal_tile, M_temporal_tile, K_spatial_tile, N_spatial_tile)
);

/////
//// OS Order : M (i) -> N (j) -> K (k) 
////  M - row of A, row of C
////  N - col of B, col of C
////  K - reduction dim
//// when tile size = 4, (N_temporal_tile, K_temporal_tile, M_spatial_tile, N_spatial_tile)
//// when tile size = 3, (K_temporal_tile, M_spatial_tile, N_spatial_tile), N_temporal_tile == 1
//// when tile size = 2, (M_spatial_tile, N_spatial_tile), N_temporal_tile == 1, K_temporal_tile = N
mlir::affine::AffineForOp TiledOutputStationaryGemm(
  OpBuilder opbuilder, ADORATensor::GemmOp op, ArrayRef<int64_t> tilesize //(K_temporal_tile, M_temporal_tile, K_spatial_tile, N_spatial_tile)
);

/// Initialize output buffer `out` with values from `C` for a logical `{M, N}`
/// GEMM result.
///
/// - If `out` and `C` have the same memref type, emits `memref.copy`.
/// - Otherwise, generates nested loops over `(m, n)` and loads `C` with
///   simple broadcast rules (scalar / vector / matrix / batch-1),
///   then stores into `out`.
///
/// Assumes:
/// - `outMN = {M, N}`
/// - `out` is `{M,N}` or `{1,M,N}`
/// - `C` follows common NN broadcast patterns.
///
/// Used in Output-Stationary GEMM lowering to initialize the accumulator.
mlir::Operation* initOutWithC2DLike(
    OpBuilder &b, Location loc,
    Value out, Value C,
    ArrayRef<int64_t> outMN);


/////////////////////////
/// Tool functions
/////////////////////////

////// Create operation functions, automatically set data type according to operands
mlir::Value getConstantOpAccordingToDataType(OpBuilder &builder, Location loc, Type datatype, float value = 0);
mlir::Operation* genArithAddOpAccordingToDataType(OpBuilder &builder, Location loc, mlir::Value lhs, mlir::Value rhs);
mlir::Operation* genArithMulOpAccordingToDataType(OpBuilder &builder, Location loc, mlir::Value lhs, mlir::Value rhs);

template <typename T> inline void setPingpongAttr(T op){
  op.getOperation()->setAttr("Pingpong", mlir::UnitAttr::get(op.getContext()));
}
inline void setPingpongAttr(mlir::Operation* op){
  op->setAttr("Pingpong", mlir::UnitAttr::get(op->getContext()));
}

static inline llvm::SmallVector<int64_t, 2> getShape(mlir::Value v) {
  mlir::Type type = v.getType();
  if (auto shapedTy = type.dyn_cast<mlir::ShapedType>()) {
    if (shapedTy.hasRank()) {
      return llvm::SmallVector<int64_t, 4>(shapedTy.getShape().begin(),
                                           shapedTy.getShape().end());
    }
  }
  return {};
}

}
}

#endif // ADORA_TENSOR_GEMM_OP_LOWER_H