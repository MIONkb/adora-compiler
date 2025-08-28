#ifndef ADORA_TENSOR_OP_LOWER_H
#define ADORA_TENSOR_OP_LOWER_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"

#include "mapper/mapper_sa.h"
#include "tensorop/TensorOpVisitor.h"

namespace mlir{
namespace ADORA{
namespace ADORATensor{

class TensorOpCDFGVisitor : public ADORATensorOpVisitorBase<TensorOpCDFGVisitor, bool> {
public:
  /// Class define
  OpBuilder opbuilder;

  /// TensorOpCDFGVisitor
  // TensorOpCDFGVisitor(){}
  TensorOpCDFGVisitor(MLIRContext* _) : opbuilder(_){}

  using ADORATensorOpVisitorBase::visitOp;


  /// Function members

  bool visitOp(ADORATensor::GemmOp op);

  bool visitInvalidOp(Operation* op) override {
    return false;
  }
}; /// end of TensorOpMapper class

}
}
}



#endif // ADORA_TENSOR_OP_LOWER_H