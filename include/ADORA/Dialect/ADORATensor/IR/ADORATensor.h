//===- ADORATensor.h - ADORATensor dialect ----------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef CGRAOPT_DIALECT_ADORATENSOR_IR_H_
#define CGRAOPT_DIALECT_ADORATENSOR_IR_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LogicalResult.h"


// #include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"

// #include "llvm/ADT/SmallSet.h" /// use std::unordered_set instead of std::list
#include<set>
//===----------------------------------------------------------------------===//
// Test Dialect
//===----------------------------------------------------------------------===//

#include "ADORA/Dialect/ADORATensor/IR/ADORATensorOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Test Dialect Operations
//===----------------------------------------------------------------------===//
#include "ADORA/Dialect/ADORATensor/Interface/ShapeInferenceOpInterface.h"
#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"

#define GET_OP_CLASSES
#include "ADORA/Dialect/ADORATensor/IR/ADORATensorOps.h.inc"

#include "ADORA/Dialect/ADORATensor/IR/ADORATensorOpsTypes.h.inc"
//===----------------------------------------------------------------------===//
// ADORA Dialect Helpers
//===----------------------------------------------------------------------===//

namespace mlir {
namespace ADORA {
namespace ADORATensor {


func::FuncOp 
  ConvertMatmulToFunc(mlir::linalg::MatmulOp op, llvm::SetVector<mlir::Value> &operands, std::string FnName);

}
} // namespace ADORA
} // namespace mlir

#endif //CGRAOPT_DIALECT_ADORA_IR_Test_H_
