//===- PassDetail.h --------------------- --------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_ADORATENSOR_TRANSFORMS_PASSDETAIL_Test_H_
#define DIALECT_ADORATENSOR_TRANSFORMS_PASSDETAIL_Test_H_

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
#include "RAAA/Dialect/ADORATensor/Transforms/Passes.h.inc"


/////// Tools
template <typename DstOp, typename SrcOp>
DstOp ConvertToSameADORATensorOp(SrcOp src) {
  OpBuilder builder(src);
  mlir::Operation* srcop = src.getOperation();
  auto loc = srcop->getLoc();

  assert(srcop->getNumOperands() == DstOp::getExpectedNumOperands());

  return builder.create<DstOp>(
    loc,
    src.getResultTypes(),   
    src.getOperands(),     
    src->getAttrs()       
  );
}

} // end namespace mlir

#endif // DIALECT_ADORA_TRANSFORMS_PASSDETAIL_Test_H_
