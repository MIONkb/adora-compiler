#ifndef ADORA_TENSOR_OP_MAP_H
#define ADORA_TENSOR_OP_MAP_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"

#include "mapper/mapper_sa.h"
#include "TensorOpVisitor.h"

#define ADORA_TENSOR_MAPPER MapperSA

namespace mlir{
namespace ADORA{

void MapAdoraTensorOp(MLIRContext* context, mlir::ModuleOp module, 
                    std::vector<ADORA_TENSOR_MAPPER*> mappers,
                    ADG* adg, int timeout_ms, int max_iters, bool objOpt);


class TensorDataflowGen : public ADORATensorOpVisitorBase<TensorDataflowGen, bool> {
public:
  /// Class define
  OpBuilder opbuilder;
  std::vector<ADORA_TENSOR_MAPPER*> mappers;

  /// TensorDataflowGen
  // TensorDataflowGen(){}
  TensorDataflowGen(MLIRContext* _) : opbuilder(_){}

  using ADORATensorOpVisitorBase::visitOp;


  /// Function members
  bool visitOp(ADORATensor::GemmOp op);

  bool visitInvalidOp(Operation* op) override {
    return false;
  }
}; /// end of TensorOpMapper class

}
}



#endif // ADORA_TENSOR_OP_MAP_H