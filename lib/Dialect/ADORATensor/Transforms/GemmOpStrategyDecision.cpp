//===----------------------------------------------------------------------===//
//
// This file implements automatic dataflow strategy decision for ADORA Tensor GemmOp
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/OperationSupport.h"
// #include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Support/LLVM.h"

#include <iostream>
#include <string>
#include <bit>

// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

// For op transformation
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "PassDetail.h"
// lower tensor op
#include "../../../../mapper/include/tensorop/TensorOp.h"


// // DFG
// ADORA::KernelOp* _kernel_toDFG;
// int _variable_config_cnt = 0;

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;

#define DEBUG_TYPE "adora-tensor-dfg-gen"

namespace mlir{
namespace ADORA{
namespace ADORATensor{

class ADORAGemmOpStrategyDecisionPass : public ADORAGemmOpStrategyDecisionPassBase<ADORAGemmOpStrategyDecisionPass>
{
  void runOnOperation() override;
};

std::unique_ptr<OperationPass<ModuleOp>> createADORAGemmOpStrategyDecisionPass()
{
  return std::make_unique<ADORAGemmOpStrategyDecisionPass>();
}

}
}
} // namespace

void ADORAGemmOpStrategyDecisionPass::runOnOperation()
{
  mlir::ModuleOp m = getOperation();
  llvm::errs() << "[test] kernel! " ; m->dump() ;

  m.walk([&](mlir::Operation* op) {

  });
  
  // assert(cnt == 1 && "There should be only 1 topFunc in IR Module.");

}