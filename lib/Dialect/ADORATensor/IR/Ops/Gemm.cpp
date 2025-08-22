//===------------------ MatMul.cpp - ADORATensor Operations ----------------------===//

#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"

using namespace mlir;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;

namespace mlir{
namespace ADORA{
namespace ADORATensor{

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//
LogicalResult ADORATensor::GemmOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // // Cannot infer shape if no shape exists.
  // if (!hasShapeAndRank(getA()) || !hasShapeAndRank(getB()))
  //   return success();

  // Type elementType = mlir::cast<ShapedType>(getA().getType()).getElementType();
  // ONNXMatMulOpShapeHelper shapeHelper(getOperation(), {});
  // return shapeHelper.computeShapeAndUpdateType(elementType);
}

}
}
}