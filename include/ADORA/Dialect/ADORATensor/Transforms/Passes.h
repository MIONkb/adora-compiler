//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef ADORATENSOR_DIALECT_PASSES_H_
#define ADORATENSOR_DIALECT_PASSES_H_

#include "mlir/Pass/Pass.h"


namespace mlir {
namespace ADORA {
namespace ADORATensor{

/// Lower some operators in Linalg dialect to 
std::unique_ptr<OperationPass<ModuleOp>> createLinalgToSystolicGEMMPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//
// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "ADORA/Dialect/ADORATensor/Transforms/Passes.h.inc"

} // namespace ADORATensor
} // namespace ADORA
} // namespace mlir

#endif // ADORATensor_DIALECT_PASSES_H_
