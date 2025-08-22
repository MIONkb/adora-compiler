//===------------------ mapGemm.cpp - ADORATensor Lower process ----------------------===//
#include "tensorop/TensorOp.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"

namespace mlir{
namespace ADORA{

bool TensorDFGGen::visitOp(ADORATensor::GemmOp op){
  
  return true;
}

}
}