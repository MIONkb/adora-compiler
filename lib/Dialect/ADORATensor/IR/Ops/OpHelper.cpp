//===------- ADORATensorOpsHelper.cpp - Helper functions for ADORATensor dialects -------===//

#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Path.h"

#include "RAAA/Dialect/ADORATensor/IR/ADORATensor.h"
// #include "src/Dialect/Mlir/IndexExpr.hpp"
// #include "src/Dialect/ONNX/DialectBuilder.hpp"
// #include "src/Dialect/ONNX/ONNXLayoutHelper.hpp"
// #include "src/Dialect/ONNX/ONNXOps.hpp"
// #include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
// #include "src/Support/TypeUtilities.hpp"

// Identity affine
using namespace mlir;

namespace mlir {
namespace ADORA{
namespace ADORATensor{


/// Test if a value is a scalar constant tensor or not, i.e. tensor<dtype> or
/// tensor<1xdtype>.
// bool isScalarConstantTensor(Value v) {
//   if (!hasShapeAndRank(v))
//     return false;

//   auto t = mlir::dyn_cast<ShapedType>(v.getType());
//   int64_t r = t.getRank();
//   return isDenseONNXConstant(v) &&
//          ((r == 0) || ((r == 1) && (t.getShape()[0] == 1)));
// }

// /// Test if 'val' has shape and rank or not.
// bool hasShapeAndRank(Value val) {
//   Type valType = val.getType();
//   ShapedType shapedType;
//   if (SeqType seqType = mlir::dyn_cast<SeqType>(valType))
//     shapedType = mlir::dyn_cast<ShapedType>(seqType.getElementType());
//   else if (OptType optType = mlir::dyn_cast<OptType>(valType))
//     shapedType = mlir::dyn_cast<ShapedType>(optType.getElementType());
//   else
//     shapedType = mlir::dyn_cast<ShapedType>(valType);
//   return shapedType && shapedType.hasRank();
// }

// bool hasShapeAndRank(Operation *op) {
//   int num = op->getNumOperands();
//   for (int i = 0; i < num; ++i)
//     if (!hasShapeAndRank(op->getOperand(i)))
//       return false;
//   return true;
// }
}
}
}
