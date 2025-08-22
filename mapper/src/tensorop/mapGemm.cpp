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

affine::AffineForOp GenerateNestedLoopWithoutLoopCarry(
    OpBuilder &builder, Location loc,
    int level = 1, SmallVector<int> Upperbounds = {1}, 
    function_ref<void(OpBuilder &, Location, Value, ValueRange)> InnerMostBodyBuilder = nullptr) 
{

  assert(level > 0 && "Loop nest level must be > 0");
  assert((int)Upperbounds.size() == level && 
         "Upperbounds size must == loop level");

  // create outermost loop 
  AffineForOp outer = builder.create<AffineForOp>(
      loc, /*lb*/ 0, /*ub*/ Upperbounds[0], /*step*/ 1);

  AffineForOp current = outer;
  // create inner loop one by one
 for (int i = 1; i < level; i++) {
    OpBuilder innerBuilder(current.getBody(),
                           std::prev(current.getBody()->end()));
    auto inner = innerBuilder.create<affine::AffineForOp>(
        loc, 0, Upperbounds[i], 1);

    current = inner;
  }


  return outer;
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
  int64_t tilerow = tilesize[0];
  int64_t tilecol = tilesize[1];

  llvm::SmallVector<int64_t, 2> ShapeA = getShape(op.getA());
  llvm::SmallVector<int64_t, 2> ShapeB = getShape(op.getB());
  llvm::SmallVector<int64_t, 2> ShapeC = getShape(op.getC());

  assert(ShapeA.size() == 2 && ShapeB.size() == 2 && ShapeC.size() == 2);
  assert(ShapeA[0] == ShapeC[0] && ShapeA[1] == ShapeB[0] && ShapeB[1] == ShapeC[1]);
  assert(ShapeA[0] % tilerow == 0 && ShapeB[1] % tilecol == 0);

  int64_t M_aftertile = ShapeA[0] / tilerow;
  int64_t N_aftertile = ShapeB[1] / tilecol;

  //////////////////////////////////////
  /// Generate systolic gemm
  //////////////////////////////////////
  OpBuilder builder(op);
  AffineForOp loop = GenerateNestedLoopWithoutLoopCarry(
    builder, op.getLoc(), 
    /*level*/3, 
    /*upper bounds*/{N_aftertile, ShapeB[0], M_aftertile} //// N -> K -> M
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