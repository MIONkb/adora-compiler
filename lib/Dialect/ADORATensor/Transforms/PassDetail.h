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
#include "ADORA/Dialect/ADORATensor/Transforms/Passes.h.inc"


/////// Tools
template <typename DstOp, typename SrcOp>
DstOp ConvertToSameADORATensorOp(SrcOp src) {
  OpBuilder builder(src);
  mlir::Operation* srcop = src.getOperation();
  auto loc = srcop->getLoc();

  //// Verify the operand number
  if(srcop->getNumOperands() != DstOp::getExpectedNumOperands()){
    src.dump();
    assert(false && "Two tensor op should get same operands number");
  }
  
  //// Build a new ADORATensor op
  DstOp dst = builder.create<DstOp>(
    loc,
    src.getResultTypes(),   
    src.getOperands(),     
    src->getAttrs()       
  );

  //// Replace the old one
  // src.getBlock()->dump();
  // src.getBlock()->push_back(dst);
  // src.getBlock()->dump();
  dst.getOperation()->moveBefore(src);
  src.getOperation()->getBlock()->dump();
  src.getOperation()->replaceAllUsesWith(dst);
  src.erase();

  return dst;
}

} // end namespace mlir

#endif // DIALECT_ADORA_TRANSFORMS_PASSDETAIL_Test_H_
