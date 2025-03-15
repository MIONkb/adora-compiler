//===- PassDetail.h --------------------- --------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_ADORA_TRANSFORMS_PASSDETAIL_Test_H_
#define DIALECT_ADORA_TRANSFORMS_PASSDETAIL_Test_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace mlir {

namespace arith {
class ArithmeticDialect;
class AffineDialect;
} // namespace Tensor
#define GEN_PASS_CLASSES
#include "RAAA/Dialect/ADORA/Transforms/Passes.h.inc"

} // end namespace mlir

#endif // DIALECT_ADORA_TRANSFORMS_PASSDETAIL_Test_H_
