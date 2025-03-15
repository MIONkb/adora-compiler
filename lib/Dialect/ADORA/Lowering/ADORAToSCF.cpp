//===- FuncToLLVM.cpp - Func to LLVM dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert ADORA dialect
// into builtin scf dialect.
//
//===----------------------------------------------------------------------===//


#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/Attributes.h"
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"

// #include <iostream>
// #include <algorithm>
// #include <functional>

#include "./LowerPassDetail.h"
#include "RAAA/Dialect/ADORA/Lowering/LowerPasses.h"
#include "RAAA/Dialect/ADORA/IR/ADORA.h"

using namespace llvm; // for llvm.errs()
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;
// using Value = mlir::Value;

#define PASS_NAME "ADORA-convert-loadstore-to-scf"

namespace {

/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
struct BlockLoadOpLowering : public OpRewritePattern<ADORA::DataBlockLoadOp> {
public:
  using OpRewritePattern<ADORA::DataBlockLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ADORA::DataBlockLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    // SmallVector<Value> indices(op.getMapOperands());
    auto resultOperands =
        ::mlir::affine::expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), op.getMapOperands());
    // llvm::errs() << "[debug] op: " << op << "\n";
    // for(auto oper : *resultOperands){
    //   llvm::errs() << "[debug] oper: " << oper << "\n";
    // }
    if (!resultOperands)
      return failure();

    // Build vector.load memref[expandedMap.results].
    // rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getMemRef(),
    //                                             *resultOperands);
    return success();
  }
};


} // namespace


namespace {
/// A pass converting Func operations into the LLVM IR dialect.
struct ConvertADORAToSCFPass
    : public ConvertADORAToSCFBase<ConvertADORAToSCFPass> {
  ConvertADORAToSCFPass() = default;
  
  SmallVector<::llvm::StringRef, 8> KernelNameVec;

  void runOnOperation() override {

    // if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
    //         this->dataLayout, [this](const Twine &message) {
    //           getOperation().emitError() << message.str();
    //         }))) {
    //   signalPassFailure();
    //   return;
    // // }

    ModuleOp m = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<BlockLoadOpLowering>(patterns.getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect>();

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();

    // m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
    //            StringAttr::get(m.getContext(), this->dataLayout));
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::ADORA::createConvertADORAToSCFPass() {
  return std::make_unique<ConvertADORAToSCFPass>();
}

