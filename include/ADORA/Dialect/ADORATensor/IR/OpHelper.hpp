//===---------- OpHelper.hpp - Helper functions for ADORATensor dialects ---------===//

#ifndef ADORATENSOR_OPS_HELPER_H
#define ADORATENSOR_OPS_HELPER_H

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "ADORATensor.h"

#include <algorithm>
#include <string>

namespace mlir {
namespace ADORA {
namespace ADORATensor {

/// Test if a value is a scalar constant tensor or not, i.e. tensor<dtype> or
/// tensor<1xdtype>.
bool isScalarConstantTensor(Value v);

/// Test if 'val' has shape and rank or not.
bool hasShapeAndRank(mlir::Value val);
bool hasShapeAndRank(mlir::Operation *op);


} } }
#endif
