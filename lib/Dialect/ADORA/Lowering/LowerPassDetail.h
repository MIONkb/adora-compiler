//===- PassDetail.h --------------------- --------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_ADORA_LOWERING_PASSDETAIL_Test_H_
#define DIALECT_ADORA_LOWERING_PASSDETAIL_Test_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "RAAA/Dialect/ADORA/IR/ADORA.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
// namespace ADORA {
#define GEN_PASS_CLASSES
#include "RAAA/Dialect/ADORA/Lowering/LowerPasses.h.inc"
// }
} // end namespace mlir

#endif // DIALECT_ADORA_LOWERING_PASSDETAIL_Test_H_
