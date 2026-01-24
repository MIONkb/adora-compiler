//===------------------ Conv.cpp - ADORATensor Operations -----------------------===//

#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"

using namespace mlir;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;

namespace mlir {
namespace ADORA {
namespace ADORATensor {

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

LogicalResult ADORATensor::ConvOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
    
  // TODO: Implement shape inference logic for Convolution.
  // Currently, we return success() to allow the compilation to pass.
  // In the future, you should calculate the output shape based on:
  //   Input Shape, Kernel Shape, Strides, Pads, Dilations

  // // Cannot infer shape if no shape exists.
  // if (!hasShapeAndRank(getA()) || !hasShapeAndRank(getB()))
  //   return success();

  // Type elementType = mlir::cast<ShapedType>(getA().getType()).getElementType();
  // ONNXMatMulOpShapeHelper shapeHelper(getOperation(), {});
  // return shapeHelper.computeShapeAndUpdateType(elementType);
  return success();
}

} // namespace ADORATensor
} // namespace ADORA
} // namespace mlir