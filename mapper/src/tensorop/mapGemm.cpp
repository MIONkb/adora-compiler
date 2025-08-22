//===------------------ mapGemm.cpp - ADORATensor Lower process ----------------------===//
/// builtin dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"

#include "tensorop/TensorOp.h"

using namespace ::mlir::ADORA::ADORATensor;
using namespace ::mlir::affine;

namespace mlir{
namespace ADORA{

static llvm::SmallVector<int64_t, 2> getShape(mlir::Value v) {
  mlir::Type type = v.getType();
  if (auto shapedTy = type.dyn_cast<mlir::ShapedType>()) {
    if (shapedTy.hasRank()) {
      return llvm::SmallVector<int64_t, 4>(shapedTy.getShape().begin(),
                                           shapedTy.getShape().end());
    }
  }
  return {};
}

using WSBodyBuilderFn = std::function<void(OpBuilder &, Location, ValueRange)>;

affine::AffineForOp GenerateNestedLoopWithoutLoopCarry(
    OpBuilder &builder, Location loc,
    int level = 1, SmallVector<int> Upperbounds = {1}, 
    WSBodyBuilderFn InnerMostBodyBuilder = nullptr) 
{

  assert(level > 0 && "Loop nest level must be > 0");
  assert((int)Upperbounds.size() == level && 
         "Upperbounds size must == loop level");

  // create outermost loop 
  AffineForOp outer = builder.create<AffineForOp>(
      loc, /*lb*/ 0, /*ub*/ Upperbounds[0], /*step*/ 1);
  AffineForOp current = outer;

  SmallVector<Value> allIvs; /// collect itervar from every level
  allIvs.push_back(current.getInductionVar());

  // create inner loop one by one
 for (int i = 1; i < level; i++) {
    OpBuilder innerBuilder(current.getBody(),
                           std::prev(current.getBody()->end()));
    if(i == level - 1){
      auto inner = innerBuilder.create<affine::AffineForOp>(
        // loc, 0, Upperbounds[i], 1, /*iterArgs =*/ ValueRange({}), InnerMostBodyBuilder);
        loc, (int64_t)0, (int64_t)Upperbounds[i], (int64_t)1, /*iterArgs =*/ ValueRange(), 
        [&](OpBuilder &b, Location loc, Value v, ValueRange vs) {
          /// v here is useless
          allIvs.push_back(v);
          InnerMostBodyBuilder(b, loc, allIvs);
        });
      current = inner;
    }
    else{
      auto inner = innerBuilder.create<affine::AffineForOp>(
        loc, 0, Upperbounds[i], 1);
        
      current = inner;
      allIvs.push_back(current.getInductionVar());
    }
  }

  return outer;
}



/// @brief Loop body builder for the innermost tiled GEMM computation 
///        under Weight-Stationary (WS) dataflow. 
///
/// This function returns a body-builder lambda of type `WSBodyBuilderFn`, 
/// which is meant to be plugged into an `affine.for` nest. 
/// Within the loop body, affine load/store operations are generated to:
///   - Load an activation element from A, indexed by (i, k),
///   - Load a weight element from B, indexed by (k, j), 
///   - Load the current partial sum from C, indexed by (i, j),
///   - Perform multiply-accumulate (A * B + C),
///   - Store the updated result back into C.
///
/// The induction variables (`ivs`) are expected to come in the order {j, k, i}, 
/// corresponding to the outer loop nest over (N, K, M). 
/// For each loop iteration, the builder further expands a `tile_row_size × tile_col_size` 
/// micro-kernel of MAC operations inside the loop body.
///
/// @param builder      The MLIR OpBuilder used to generate operations.
/// @param A            MemRef value representing the activation/input tensor.
/// @param B            MemRef value representing the weight tensor (stationary).
/// @param C            MemRef value representing the output/accumulation tensor.
/// @param tile_row_size Number of rows in the tile (micro-kernel row dimension).
/// @param tile_col_size Number of columns in the tile (micro-kernel column dimension).
///
/// Example pseudocode for one tile iteration:
/// ```text
/// for j in N
///   for k in K
///     for i in M
///       for n in tile_row_size
///         for t in tile_col_size
///           C[i, j+n] += A[i, k+t] * B[k+t, j+n]
WSBodyBuilderFn InnermostBodyOfTiledWithWeightStationary(
    OpBuilder builder,
    mlir::Value A,
    mlir::Value B,
    mlir::Value C,
    int tile_row_size,
    int tile_col_size
) {
return [=](OpBuilder &builder, Location loc, ValueRange ivs) {
    // ivs = {j k i}
    assert(ivs.size() == 3 && "Expect 3 loop induction variables (i,j,k)");
    Value j_it = ivs[0];
    Value k_it = ivs[1];
    Value i_it = ivs[2];

    j_it.dump();
    k_it.dump();
    i_it.dump();

    auto i32Type = builder.getI32Type();

    /**
     * Example: N K M (j k i)
     * affine.for %arg3 = 0 to 9 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 36 {
            %0 = affine.load %arg0[%arg5, %arg4] : memref<?x36xi32>
            %1 = affine.load %arg1[%arg4, %arg3] : memref<?x36xi32>
            %2 = arith.muli %0, %1 : i32
            %3 = affine.load %arg2[%arg5, %arg3] : memref<?x36xi32>
            %4 = arith.addi %3, %2 : i32
            affine.store %4, %arg2[%arg5, %arg3] : memref<?x36xi32>
            ...
          }
        }
      }
     */
    for (int n = 0; n < tile_row_size; ++n) {
      for (int k = 0; k < tile_col_size; ++k) {
        /// affine map of A : 
        /// %0 = affine.load %arg0[%arg5, %arg4] : memref<?x36xi32>
        /// A[i, k]
        AffineExpr rowExpr_A = builder.getAffineDimExpr(0);
        AffineExpr colExpr_A = builder.getAffineDimExpr(1) + k;
        AffineMap map_A = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                  {rowExpr_A, colExpr_A}, builder.getContext());
        AffineLoadOp LoadA = builder.create<affine::AffineLoadOp>(
            loc, A, map_A, ValueRange{i_it, k_it});

        /// affine map of B : stationary
        /// %1 = affine.load %arg1[%arg4, %arg3] : memref<?x36xi32>
        /// B[k, j]
        AffineExpr rowExpr_B = builder.getAffineDimExpr(0) + k;
        AffineExpr colExpr_B = builder.getAffineDimExpr(1) + n;
        AffineMap map_B = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                  {rowExpr_B, colExpr_B}, builder.getContext());
        AffineLoadOp LoadB = builder.create<affine::AffineLoadOp>(
            loc, B, map_B, ValueRange{k_it, j_it});

        /// affine map of C
        /// %3 = affine.load %arg2[%arg5, %arg3] : memref<?x36xi32>
        /// C[i, j]
        AffineExpr rowExpr_C = builder.getAffineDimExpr(0);
        AffineExpr colExpr_C = builder.getAffineDimExpr(1) + n;
        AffineMap map_C = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                  {rowExpr_C, colExpr_C}, builder.getContext());
        AffineLoadOp LoadC = builder.create<affine::AffineLoadOp>(
            loc, C, map_C, ValueRange{i_it, j_it});

        // Mul AxB

        // mul
        Value mulVal = builder.create<arith::MulIOp>(loc, LoadA, LoadB);

        // add
        Value add = builder.create<arith::AddIOp>(loc, mulVal, LoadC);

        // store back to C
        AffineStoreOp StoreC = builder.create<affine::AffineStoreOp>(
            loc, add, C, map_C, ValueRange{i_it, j_it});
        
        LoadC.getOperation()->getBlock()->dump();
      }
    }
  };
}

/////
//// WS Order : N (j) -> K (k) -> M (i)
////  M - row of A, row of C
////  N - col of B, col of C
////  K - reduction dim
void TiledWeightStationaryGemm(ADORATensor::GemmOp op, ArrayRef<int64_t> tilesize){
  assert(tilesize.size() == 2);

  //////////////////////////////////////
  /// Get tiled matmul parameter
  //////////////////////////////////////
  // int64_t tilerow = tilesize[0];
  // int64_t tilecol = tilesize[1];

  int64_t tilerow = 1;
  int64_t tilecol = 1;

  llvm::SmallVector<int64_t, 2> ShapeA = getShape(op.getA());
  llvm::SmallVector<int64_t, 2> ShapeB = getShape(op.getB());
  llvm::SmallVector<int64_t, 2> ShapeC = getShape(op.getC());

  assert(ShapeA.size() == 2 && ShapeB.size() == 2 && ShapeC.size() == 2);
  assert(ShapeA[0] == ShapeC[0] && ShapeA[1] == ShapeB[0] && ShapeB[1] == ShapeC[1]);
  assert(ShapeA[0] % tilerow == 0 && ShapeB[1] % tilecol == 0);

  int64_t M_aftertile = ShapeA[0] / tilerow;
  int64_t N_aftertile = ShapeB[1];
  int64_t K_aftertile = ShapeB[0] / tilecol;

  //////////////////////////////////////
  /// Generate systolic gemm
  //////////////////////////////////////
  OpBuilder builder(op);
  AffineForOp loop = GenerateNestedLoopWithoutLoopCarry(
    builder, op.getLoc(), 
    /*level*/3, 
    /*upper bounds*/{N_aftertile, K_aftertile, M_aftertile}, //// N -> K -> M
    /*InnerMostBodyBuilder*/InnermostBodyOfTiledWithWeightStationary(
      builder, op.getA(), op.getB(), op.getC(), tilerow, tilecol
    )
  );

  loop.dump();
}


bool TensorDFGGen::visitOp(ADORATensor::GemmOp op){
  SystolicImplInterface SystolicPara(op);
  ArrayRef<int64_t> tilesize = SystolicPara.getTileSize();
  StringRef stragegy = SystolicPara.getStationaryKind();
  if(stragegy == getMethodStrRef(MatMulStrategy::WeightStationary)){
    TiledWeightStationaryGemm(op, tilesize);
  }
  return true;
}

}
}