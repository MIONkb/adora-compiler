//===----------------------------------------------------------------------===//
//
// This file implements Data flow graph generation for ADORA Tensor
//
//===----------------------------------------------------------------------===//

/// builtin dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Misc/DFG.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOpLowerToKernel.h"
// #include "PassDetail.h"

// lower tensor op
#include "../../../../mapper/include/tensorop/TensorOp.h"
#include "../../../../mapper/include/tensorop/MapGemm.h"

// // DFG
// ADORA::KernelOp* _kernel_toDFG;
// int _variable_config_cnt = 0;

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;

#define DEBUG_TYPE "adora-tensor-op-lower"

namespace mlir{
namespace ADORA{
namespace ADORATensor{


bool TensorOpCDFGVisitor::visitOp(ADORATensor::GemmOp op){
  SystolicImplInterface SystolicPara(op);
  ArrayRef<int64_t> tilesize = SystolicPara.getTileSize();
  StringRef stragegy = SystolicPara.getStationaryKind();
  AffineForOp newfor;
  if(stragegy == getMethodStrRef(MatMulStrategy::WeightStationary)){
    newfor = TiledWeightStationaryGemm(opbuilder, op, tilesize); 
  }
  else if(stragegy == getMethodStrRef(MatMulStrategy::InputStationary)){
    newfor = TiledInputStationaryGemm(opbuilder, op, tilesize); 
  }
  else if(stragegy == getMethodStrRef(MatMulStrategy::OutputStationary)){
    newfor = TiledOutputStationaryGemm(opbuilder, op, tilesize); 
  }

  return true;
}

}
}
}