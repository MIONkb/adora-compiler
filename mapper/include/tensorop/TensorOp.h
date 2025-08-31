#ifndef ADORA_TENSOR_OP_MAP_H
#define ADORA_TENSOR_OP_MAP_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"

#include "emit/EmitCGRACall.h"
#include "emit/EmitPytest.h"
#include "mapper/mapper_sa.h"
#include "TensorOpVisitor.h"

#define ADORA_TENSOR_MAPPER MapperSA



namespace mlir{
namespace ADORA{
#pragma once
inline int tensorOpCnt = 0;



void MapAdoraTensorOp(MLIRContext* context, mlir::ModuleOp module, 
                    std::vector<ADORA_TENSOR_MAPPER*> mappers,
                    CGRACallEmitter* CEmitter, PytestEmitter* PyEmitter,
                    ADG* adg, std::string& OpNameFile_str,
                    int timeout_ms, int max_iters, bool objOpt);


class TensorDataflowGen : public ADORATensorOpVisitorBase<TensorDataflowGen, bool> {
public:
  /// Class define
  OpBuilder opbuilder;
  CGRACallEmitter* cEmitter;
  PytestEmitter* pyEmitter;


  // mapping args
  std::vector<ADORA_TENSOR_MAPPER*> mappers;
  ADG* _adg = nullptr;
  std::string _OpNameFile_str = "";
  int _timeout_ms;
  int _max_iters; 
  bool _objOpt;
  void setMappingArgs(ADG* adg, std::string& OpNameFile_str,
    int timeout_ms, int max_iters, bool objOpt){
    _adg = adg;
    _OpNameFile_str = OpNameFile_str;
    _timeout_ms = timeout_ms;
    _max_iters = max_iters;
    _objOpt = objOpt;
  }

  /// TensorDataflowGen
  // TensorDataflowGen(){}
  TensorDataflowGen(MLIRContext* _) : opbuilder(_){}

  using ADORATensorOpVisitorBase::visitOp;


  /// Function members
  void MapNestedForOrKernel(ADORA_TENSOR_MAPPER* mapper, 
    mlir::Operation* forOrKernel, std::string& OpNameFile_str);
  void setEmitter(CGRACallEmitter* _) {cEmitter = _;}
  void setEmitter(PytestEmitter* _) {pyEmitter = _;}

  bool visitOp(ADORATensor::GemmOp op);

  bool visitInvalidOp(Operation* op) override {
    return false;
  }
}; /// end of TensorOpMapper class

}
}



#endif // ADORA_TENSOR_OP_MAP_H