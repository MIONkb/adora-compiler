//===- ADORATensorFunctionsToKernelPass.cpp - Extract kernels from tensor functions ---===//
//
// Convert affine for-loops inside tensor functions (converted from ONNX) into
// ADORA::KernelOp, guided by function attributes such as `adora_kernel` and
// `onnx_layer`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

// #include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORATensor/Utility/Utility.h"
#include "ADORA/Dialect/ADORATensor/Transforms/Passes.h"
#include "PassDetail.h"

using namespace llvm; // for llvm::dbgs / errs
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;

#define DEBUG_TYPE "adora-extract-kernels-in-tensor-function"

namespace mlir {
namespace ADORA {
namespace ADORATensor {

#define IsKernelizableLayer(type, target) ((type) == (target))

static bool isAllowedOpInKernelBody(Operation *op) {
  auto name = op->getName().getStringRef();
  return
      name == AffineLoadOp::getOperationName() ||
      name == AffineStoreOp::getOperationName() ||
      name == AffineForOp::getOperationName() ||
      name == AffineIfOp::getOperationName() ||
      name == AffinePrefetchOp::getOperationName() ||
      name == AffineVectorLoadOp::getOperationName() ||
      name == AffineVectorStoreOp::getOperationName() ||
      name == AffineYieldOp::getOperationName() ||
      name == mlir::memref::LoadOp::getOperationName() ||
      // arith
      name == mlir::arith::TruncFOp::getOperationName() ||
      name == mlir::arith::TruncIOp::getOperationName() ||
      name == mlir::arith::UIToFPOp::getOperationName();
}

static LogicalResult extractSingleForToKernel(AffineForOp forOp) {
  LLVM_DEBUG(forOp.dump());

  auto walkResult = forOp.walk([&](Operation *op) {
    if (isAllowedOpInKernelBody(op))
      return WalkResult::advance();
    return WalkResult::interrupt();
  });

  if (walkResult.wasInterrupted()) {
    if (succeeded(ADORA::SpecifiedAffineFortoKernel(forOp)))
      return success();
  }

  return failure();
}

static void renameKernelsInFunc(func::FuncOp func) {
  SmallVector<ADORA::KernelOp, 4> kernels;
  func.walk([&](ADORA::KernelOp kn) { kernels.push_back(kn); });

  if (kernels.empty())
    return;

  StringRef funcName = func.getSymName();

  if (kernels.size() == 1) {
    kernels.front().setKernelName(funcName.str());
    return;
  }

  unsigned kernelCount = 0;
  for (ADORA::KernelOp kn : kernels) {
    std::string kernelName =
        (Twine(funcName) + "_" + Twine(kernelCount++)).str();
    kn.setKernelName(kernelName);
  }
}

static LogicalResult convertAllLoopsToKernels(func::FuncOp func,
                                              ArrayRef<AffineForOp> loops) {
  bool changed = false;
  SmallVector<AffineForOp, 4> loopsCopy(loops.begin(), loops.end());

  for (AffineForOp forOp : loopsCopy) {
    if (!forOp || forOp->getParentOfType<func::FuncOp>() != func)
      continue;

    if (succeeded(extractSingleForToKernel(forOp)))
      changed = true;
  }

  if (changed)
    renameKernelsInFunc(func);

  return success();
}


static LogicalResult handleMultiLoopFunction(func::FuncOp func,
                                             ArrayRef<AffineForOp> loops) {
  LLVM_DEBUG({
    dbgs() << "TODO handle multi-loop tensor function `"
           << func.getSymName() << "` with " << loops.size()
           << " affine.for ops\n";
  });

  return success();
}

struct ADORATensorFunctionsToKernel
    : public ADORATensorFunctionsToKernelBase<ADORATensorFunctionsToKernel> {
  ADORATensorFunctionsToKernel() = default;
  using Base = ADORATensorFunctionsToKernelBase<ADORATensorFunctionsToKernel>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (!func->hasAttrOfType<UnitAttr>("adora_kernel"))
      return;

    SmallVector<AffineForOp, 4> loops;
    for (Operation &op : llvm::make_early_inc_range(func.getOps())) {
      if (auto forOp = dyn_cast<AffineForOp>(&op)) {
        loops.push_back(forOp); 
      }
    }
    if (loops.empty())
      return;

    StringRef onnxLayer;
    if (auto layerAttr = func->getAttrOfType<StringAttr>("onnx_layer"))
      onnxLayer = layerAttr.getValue();

    LLVM_DEBUG({
      dbgs() << "[ADORATensorFunctionsToKernel] func `"
             << func.getSymName() << "` has " << loops.size()
             << " affine.for loops, onnx_layer = \"" << onnxLayer << "\"\n";
    });

    if (  IsKernelizableLayer(onnxLayer, "Mul") 
        ||IsKernelizableLayer(onnxLayer, "Add") 
        ||IsKernelizableLayer(onnxLayer, "MatMul")  ) {
      if (failed(convertAllLoopsToKernels(func, loops))) {
        signalPassFailure();
      }
      return;
    }

    if (failed(handleMultiLoopFunction(func, loops))) {
      signalPassFailure();
      return;
    }
  }
};


std::unique_ptr<OperationPass<func::FuncOp>>
createADORATensorFunctionsToKernelPass() {
  return std::make_unique<ADORATensorFunctionsToKernel>();
}

}
}
} // namespace