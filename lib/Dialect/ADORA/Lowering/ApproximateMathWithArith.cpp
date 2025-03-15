//===- FuncToLLVM.cpp - Func to LLVM dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert complex math operations to a combination of
// simple operations.
//
//===----------------------------------------------------------------------===//


#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

#include <iostream>
#include <algorithm>
#include <functional>

#include "./LowerPassDetail.h"
#include "RAAA/Dialect/ADORA/Lowering/LowerPasses.h"
#include "RAAA/Dialect/ADORA/IR/ADORA.h"


// using namespace llvm; // for llvm.errs()
using namespace mlir;
using namespace mlir::ADORA;
// using Value = mlir::Value;

#define PASS_NAME "ADORA-approximate-math"


namespace {
class RsqrtConverter : public OpRewritePattern<math::RsqrtOp> {
  /**=================================
   * Convert math.rsqrt with following algorithm:
   * 
   * float Q_rsqrt( float number )
  {
	  long i;
	  float x2, y;
	  const float threehalfs = 1.5F;

	  x2 = number * 0.5F;
	  y  = number;
	  i  = * ( long * ) &y;                       // evil floating point bit level hacking
	  i  = 0x5f3759df - ( i >> 1 );               // what the fuck? 
	  y  = * ( float * ) &i;
	  y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
    //	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	  return y;
  }
   * 
   * 
  ===================================*/
  
public:
  using OpRewritePattern<math::RsqrtOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::RsqrtOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();

    Type F32type = rewriter.getF32Type();
    Type I32type = rewriter.getI32Type();

    Value cst_1_2 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.5));
    Value x2 = rewriter.create<arith::MulFOp>(loc, op.getOperand(), cst_1_2);

    Value i0 = rewriter.create<arith::BitcastOp>(loc, I32type, op.getOperand());
    Value cst_1 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(I32type, 1));
    Value i1 = rewriter.create<arith::ShRUIOp>(loc, i0, cst_1);
    Value cst_magic = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(I32type, 0x5f3759df));
    Value i2 = rewriter.create<arith::SubIOp>(loc, cst_magic, i1);
    Value f0 = rewriter.create<arith::BitcastOp>(loc, F32type, i2);

    Value cst_3_2 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.5));
    Value f1 = rewriter.create<arith::MulFOp>(loc, f0, f0);
    Value f2 = rewriter.create<arith::MulFOp>(loc, f1, x2);
    Value f3 = rewriter.create<arith::SubFOp>(loc, cst_3_2, f2);
    Value result = rewriter.create<arith::MulFOp>(loc, f3, f1);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Similar but different with ArithExpand pass
template <typename OpTy, arith::CmpFPredicate pred>
struct MaxMinFOpConverter : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    Location loc = op.getLoc();
    // If any operand is NaN, 'cmp' will be true (and 'select' returns 'lhs').
    static_assert(pred == arith::CmpFPredicate::UGT ||
                      pred == arith::CmpFPredicate::ULT,
                  "pred must be either UGT or ULT");
    Value cmp = rewriter.create<arith::CmpFOp>(loc, pred, lhs, rhs);
    // Value select = rewriter.create<arith::SelectOp>(loc, cmp, lhs, rhs);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, cmp, lhs, rhs);

    // Different with ArithExpand pass
    // Do not handle the case where rhs is NaN: 'isNaN(rhs) ? rhs : select'.
    // Value isNaN = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNO,
    //                                              rhs, rhs);
    // rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isNaN, rhs, select);
    return success();
  }
};

/// A pass to lower math operations 
struct ApproximateMathWithArithPass
    : public ApproximateMathWithArithBase<ApproximateMathWithArithPass> {
  ApproximateMathWithArithPass() = default;

  void runOnOperation() override {

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    // target.addIllegalOp<tosa::ConstOp>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<
      arith::MaximumFOp,
      arith::MinimumFOp
    >();

    patterns.add<RsqrtConverter>(patterns.getContext());

    patterns.add<
      MaxMinFOpConverter<arith::MaximumFOp, arith::CmpFPredicate::UGT>,
      MaxMinFOpConverter<arith::MinimumFOp, arith::CmpFPredicate::ULT>
    >(patterns.getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::ADORA::createApproximateMathWithArithPass() {
  return std::make_unique<ApproximateMathWithArithPass>();
}

