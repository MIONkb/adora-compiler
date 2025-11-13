//===- Test.h - Test dialect --------------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef CGRAOPT_DIALECT_ADORA_KERNELOP_H_
#define CGRAOPT_DIALECT_ADORA_KERNELOP_H_

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
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// #include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"

// #include "llvm/ADT/SmallSet.h" /// use std::unordered_set instead of std::list
#include<set>

#ifndef INCLUDE_DEFINE_ADORA_DIALECT
#define INCLUDE_DEFINE_ADORA_DIALECT
#include "ADORA/Dialect/ADORA/IR/ADORAOpsDialect.h.inc"
#endif

#define GET_OP_CLASSES
#include "ADORA/Dialect/ADORA/IR/KernelOp/ADORAKernelOp.h.inc"
#include "ADORA/Dialect/ADORA/IR/KernelOp/ADORAKernelOpTypes.h.inc"
//===----------------------------------------------------------------------===//
// ADORA Dialect Helpers
//===----------------------------------------------------------------------===//

namespace mlir {
namespace ADORA {

} // namespace ADORA
} // namespace mlir

#endif //CGRAOPT_DIALECT_ADORA_KERNELOP_H_
