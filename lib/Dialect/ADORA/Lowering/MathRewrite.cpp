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

#define PASS_NAME "ADORA-math-rewrite"


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

class TanhConverter : public OpRewritePattern<math::TanhOp> {
  /**=================================
   * Convert math.tanh with gaussian continued fraction algorithm:
   * 
   * float approx_tanh_by_continues_fraction(float x)
{
    float s = x * x;
    float y = 9 + s / 11;
    y = 7 + s / y;
    y = 5 + s / y;
    y = 3 + s / y;
    y = 1 + s / y;
    y = x / y;
    return y;
}
   * 
   * 
  ===================================*/
  
public:
  using OpRewritePattern<math::TanhOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::TanhOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();

    Type F32type = rewriter.getF32Type();
    Type I32type = rewriter.getI32Type();
    
    /// s = x * x
    Value s = rewriter.create<arith::MulFOp>(loc, op.getOperand(), op.getOperand());

    /// y0 = 9 + s / 11
    Value cst_11 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(11));
    Value cst_9 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(9));
    Value d0 = rewriter.create<arith::DivFOp>(loc, s, cst_11);
    Value y0 = rewriter.create<arith::AddFOp>(loc, cst_9, d0);

    /// y1 = 7 + s / y0
    Value cst_7 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(7));
    Value d1 = rewriter.create<arith::DivFOp>(loc, s, y0);
    Value y1 = rewriter.create<arith::AddFOp>(loc, cst_7, d1);

    /// y2 = 5 + s / y1
    Value cst_5 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(5));
    Value d2 = rewriter.create<arith::DivFOp>(loc, s, y1);
    Value y2 = rewriter.create<arith::AddFOp>(loc, cst_5, d2);
  
    /// y3 = 3 + s / y2
    Value cst_3 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(3));
    Value d3 = rewriter.create<arith::DivFOp>(loc, s, y2);
    Value y3 = rewriter.create<arith::AddFOp>(loc, cst_3, d3);

    /// y4 = 1 + s / y3
    Value cst_1 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1));
    Value d4 = rewriter.create<arith::DivFOp>(loc, s, y3);
    Value y4 = rewriter.create<arith::AddFOp>(loc, cst_1, d4);

    /// y5 = x / y4
    Value result = rewriter.create<arith::DivFOp>(loc, op.getOperand(), y4);

    rewriter.replaceOp(op, result);
    return success();
  }
};

class ExpConverter : public OpRewritePattern<math::ExpOp> {
  /**=================================
   * Convert math.expOp with following algorithm:
   * 
   * float expf_fast(float x)
{
	float yf = 12102203 * x;
	int yi = (int)yf + 1064872507;

	return (*(float*)(&yi));
}


for double:

double expd_fast(double x)
{
	double yf = 0.0;
	int yi = (int)(1512775 * x) + 1072633159;
	*((int*)(&yf) + 1) = yi; // fill high-32bits of double

	return yf;
}

url:https://zhuanlan.zhihu.com/p/16418642051
   * 
   * 
  ===================================*/
  
public:
  using OpRewritePattern<math::ExpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();

    Type F32type = rewriter.getF32Type();
    Type I32type = rewriter.getI32Type();

    /// float yf = 12102203 * x;
    Value cst_12102203 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(12102203));
    Value yf = rewriter.create<arith::MulFOp>(loc, op.getOperand(), cst_12102203);

    /// int yi = (int)(1512775 * x) + 1072633159;
    Value cst_1512775 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1512775));
    Value xx = rewriter.create<arith::MulFOp>(loc, op.getOperand(), cst_1512775);
    Value xx_int = rewriter.create<arith::BitcastOp>(loc, I32type, xx);
    Value cst_1072633159 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(I32type, 1072633159));
    Value yi = rewriter.create<arith::AddIOp>(loc, xx_int, cst_1072633159);

    /// return (*(float*)(&yi));
    Value result = rewriter.create<arith::BitcastOp>(loc, F32type, yi);

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
struct MathRewritePass
    : public MathRewriteBase<MathRewritePass> {
      MathRewritePass() = default;

  void runOnOperation() override {

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    // target.addIllegalOp<tosa::ConstOp>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<
      arith::MaximumFOp,
      arith::MinimumFOp
    >();

    patterns.add< 
      RsqrtConverter, 
      TanhConverter,
      ExpConverter
    >(patterns.getContext());
    

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

std::unique_ptr<OperationPass<ModuleOp>> mlir::ADORA::createMathRewritePass() {
  return std::make_unique<MathRewritePass>();
}

