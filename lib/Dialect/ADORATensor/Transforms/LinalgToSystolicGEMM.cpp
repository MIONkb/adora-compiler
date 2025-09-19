//===- LinalgToSystolicGEMM.cpp - conversion from Linalg named operators to systolic gemm --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Transforms/Passes.h"

#include "PassDetail.h"

#define DEBUG_TYPE "ADORA-linalg-to-systolic-gemm"

namespace mlir {
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;

namespace mlir {
namespace ADORA {
namespace ADORATensor {

struct LinalgToSystolicGEMMPass
    : public LinalgToSystolicGEMMPassBase<LinalgToSystolicGEMMPass> {
public:
  int matmul_idx = 0;
  mlir::ModuleOp _m;

  void runOnOperation() override {
    // lowerLinalgToLoopsImpl<affine::AffineForOp>(getOperation());
    _m = getOperation();
    _m.walk([&](linalg::MatmulOp matmul) {
      func::FuncOp f = ConvertMatmulToSystolic(matmul);
      _m.getBodyRegion().front().push_back(f);
      _m.dump();
      return; 
    });
    
  };
  func::FuncOp ConvertMatmulToSystolic(linalg::MatmulOp matmul);
}; // struct LinalgToSystolicGEMMPass

func::FuncOp LinalgToSystolicGEMMPass::ConvertMatmulToSystolic(linalg::MatmulOp matmul){
  llvm::SetVector<mlir::Value> operands;
  for(auto operand : matmul.getOperands()){
    operands.insert(operand);
  }
  func::FuncOp func = ConvertMatmulToFunc(matmul, operands, "matmul_" + std::to_string(matmul_idx));
  
  //// Annotate the function as a systolic gemm
  

  //// lower matmul to gemm
  linalg::MatmulOp ToLowerOp = dyn_cast<linalg::MatmulOp>(func.getBody().front().front());
  ADORATensor::GemmOp newGemm;
  newGemm = ConvertToSameADORATensorOp<ADORATensor::GemmOp>(ToLowerOp);

  //// set to a 4x4 weight stationary
  ADORATensor::SystolicImplInterface Sinterface(newGemm);
  // mlir::SmallVector<int64_t> tile = {4,4};
  Sinterface.setStationaryKind(MatMulStrategy::WeightStationary);
  Sinterface.setTileSize(ArrayRef<int64_t>({4,4}));

  matmul_idx++;
  return func;
}


std::unique_ptr<OperationPass<ModuleOp>> createLinalgToSystolicGEMMPass() {
  return std::make_unique<LinalgToSystolicGEMMPass>();
}

}
} // namespace
} // namespace

// std::unique_ptr<OperationPass<ModuleOp>> ::mlir::ADORA::createLinalgToSystolicGEMMPass() {
//   return std::make_unique<LinalgToSystolicGEMMPass>();
// }
