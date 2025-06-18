//===- Passes.h - ADORA-opt pass entry points --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef ADORA_MISC_PASSES_H
#define ADORA_MISC_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir


namespace mlir {
namespace ADORA {

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//
std::unique_ptr<mlir::Pass> createTestPrintOpNestingPass();

//===----------------------------------------------------------------------===//
// Lowerings
//===----------------------------------------------------------------------===//

// TODO: Move this pass out of the Misc directory into the Conversion directory
/// Perform lowering from std operations to LLVM dialect.
/// Exposing the options of barePtrCallConv or emitCWrappers without the need
/// to know the mlir context during pass (pipeline) creation. MLIR context is
/// obtained at runtime.
///
/// This pass is based on:
///    llvm-project/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp

//===----------------------------------------------------------------------===//
// Register passes
//===----------------------------------------------------------------------===//

/// Include the auto-generated definitions for passes
// TODO: only the registration call is necessary. Move pass class decls to
// another file
#define GEN_PASS_CLASSES
#include "RAAA/Misc/ADORAMiscPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "RAAA/Misc/ADORAMiscPasses.h.inc"

} // namespace ADORA
} // namespace mlir

#endif // ADORA_MISC_PASSES_H
