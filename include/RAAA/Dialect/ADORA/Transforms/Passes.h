//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef ADORA_DIALECT_PASSES_H_
#define ADORA_DIALECT_PASSES_H_

#include "mlir/Pass/Pass.h"


namespace mlir {
namespace ADORA {

// CDFG generate Pass
std::unique_ptr<OperationPass<ModuleOp>> createADORALoopCdfgGenPass();
std::unique_ptr<OperationPass<func::FuncOp>> createExtractAffineForToKernelPass();
std::unique_ptr<OperationPass<ModuleOp>> createAdjustKernelMemoryFootprintPass();
std::unique_ptr<OperationPass<ModuleOp>> createExtractKernelToFuncPass();
std::unique_ptr<OperationPass<ModuleOp>> createSimplifyAffineLoopLevelsPass();
std::unique_ptr<OperationPass<ModuleOp>> createADORAAffineLoopUnrollPass();
std::unique_ptr<OperationPass<ModuleOp>> createAutoDesignSpaceExplorePass();
std::unique_ptr<OperationPass<func::FuncOp>> createSimplifyLoadStoreInLoopNestPass();
std::unique_ptr<OperationPass<func::FuncOp>> createAffineLoopReorderPass();
std::unique_ptr<OperationPass<ModuleOp>> createADORALoopUnrollAndJamPass();
std::unique_ptr<OperationPass<ModuleOp>> createADORAAutoUnrollPass();
std::unique_ptr<OperationPass<ModuleOp>> createScheduleADORATasksPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//
// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "RAAA/Dialect/ADORA/Transforms/Passes.h.inc"


} // namespace ADORA
} // namespace mlir

#endif // ADORA_DIALECT_PASSES_H_
