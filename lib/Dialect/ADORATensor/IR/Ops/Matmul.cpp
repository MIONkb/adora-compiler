//===------------------ MatMul.cpp - ADORATensor Operations ----------------------===//

#include "RAAA/Dialect/ADORATensor/IR/ADORATensor.h"

using namespace mlir;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;

namespace mlir{
namespace ADORA{
namespace ADORATensor{

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//
StringRef MatMulOp::getMethodStrRef(MatMulStrategy method) {
  switch (method) {
    case MatMulStrategy::WeightStationary:
      return "WeightStationary";
      break;
    case MatMulStrategy::InputStationary:
      return "InputStationary";
      break;
    case MatMulStrategy::OutputStationary:
      return "OutputStationary";
      break;
    }
}

void MatMulOp::setStationaryKind(MatMulStrategy method){
  StringAttr methodAttr = StringAttr::get(getOperation()->getContext(), getMethodStrRef(method));
  getOperation()->setAttr(getStationaryKindAttrStr(), methodAttr);
}

StringRef MatMulOp::getStationaryKind(){
  StringRef methodAttr = getOperation()->getAttr(getStationaryKindAttrStr())
      .cast<StringAttr>().strref();
  return methodAttr;
}
// LogicalResult ADORATensor::MatMulOp::inferShapes(
//     std::function<void(Region &)> doShapeInference) {
//   // Cannot infer shape if no shape exists.
//   if (!hasShapeAndRank(getA()) || !hasShapeAndRank(getB()))
//     return success();

//   Type elementType = mlir::cast<ShapedType>(getA().getType()).getElementType();
//   ONNXMatMulOpShapeHelper shapeHelper(getOperation(), {});
//   return shapeHelper.computeShapeAndUpdateType(elementType);
// }
}
}
}