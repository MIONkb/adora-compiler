//===------------------ mapGemm.cpp - ADORATensor Lower process ----------------------===//
/// builtin dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"

#include "tensorop/TensorOp.h"
#include "tensorop/Gemm/MapGemm.h"

using namespace ::mlir::ADORA::ADORATensor;
using namespace ::mlir::affine;

namespace mlir{
namespace ADORA{

bool TensorDataflowGen::visitOp(ADORATensor::GemmOp op){
  SystolicImplInterface SystolicPara(op);
  ArrayRef<int64_t> tilesize = SystolicPara.getTileSize();
  StringRef stragegy = SystolicPara.getStationaryKind();
  if(stragegy == getMethodStrRef(MatMulStrategy::WeightStationary)){
    AffineForOp newfor = TiledWeightStationaryGemm(opbuilder, op, tilesize); 
  }
  return true;
}

}
}