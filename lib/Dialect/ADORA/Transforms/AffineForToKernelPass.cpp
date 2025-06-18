//===- SCFToKernelPass.cpp - Convert a loop nest to a kernel to be accelerated -----------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Parser/Parser.h"
#include <iostream>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "RAAA/Dialect/ADORA/IR/ADORA.h"
#include "RAAA/Dialect/ADORA/Transforms/Passes.h"
#include "./PassDetail.h"


using namespace llvm; // for llvm.errs()
using namespace mlir;
using namespace mlir::ADORA;
using namespace mlir::affine;

//===----------------------------------------------------------------------===//
// AffineToKERNEL
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "adora-extract-for-to-kernel"

namespace {
  
// A pass that traverses top-level loops in the function and converts them to
// ADORA Kernel operations.  
struct AffineForKernelCaptor: public ExtractAffineForToKernelBase<AffineForKernelCaptor> {
  AffineForKernelCaptor() = default;
  void runOnOperation() override {
    if(FunctionName!="-"){
      /// function name is specified
      llvm::SmallVector<std::string> functions;
      std::string token;
      std::stringstream ss(FunctionName);
      while (std::getline(ss, token, ',')) {
        functions.push_back(token); 
      }
      std::string ThisFunction = getOperation().getSymName().str();

      if(findElement(functions, ThisFunction) == -1){
        /// this function is not specified
        return ;
      }
    }

    unsigned kernel_count = 0;
    for (Operation &op : llvm::make_early_inc_range(getOperation().getOps())) {
      if (auto forOp = dyn_cast<AffineForOp>(&op)) {
        LLVM_DEBUG(forOp.dump());
        auto ForWalkResult = forOp.walk([&](Operation *op){ 
          if(
              op->getName().getStringRef()== AffineLoadOp::getOperationName() ||
              op->getName().getStringRef()== AffineStoreOp::getOperationName()||
              op->getName().getStringRef()== AffineForOp::getOperationName()||
              op->getName().getStringRef()== AffineIfOp ::getOperationName()||
              op->getName().getStringRef()== AffinePrefetchOp ::getOperationName()||
              op->getName().getStringRef()== AffineVectorLoadOp ::getOperationName()||
              op->getName().getStringRef()== AffineVectorStoreOp ::getOperationName()||
              op->getName().getStringRef()== AffineYieldOp ::getOperationName() ||
              op->getName().getStringRef()== mlir::memref::LoadOp ::getOperationName()||
              /// arith
              op->getName().getStringRef()== mlir::arith::TruncFOp ::getOperationName() ||
              op->getName().getStringRef()== mlir::arith::TruncIOp ::getOperationName() ||
              op->getName().getStringRef()== mlir::arith::UIToFPOp ::getOperationName() )
          {
            return WalkResult::advance();
          }
          else
            return WalkResult::interrupt();
        });
      
        if(ForWalkResult.wasInterrupted()){
          if(mlir::succeeded(ADORA::SpecifiedAffineFortoKernel(forOp)))
            kernel_count++;
        }
      }
    }
    

    /// rename kernels
    if(kernel_count==1){
        /** Initialize name of kernel **/
        ADORA::KernelOp kn = *(getOperation().getOps<ADORA::KernelOp>().begin());
        std::string kernelname = getOperation().getSymName().str();
        kn.setKernelName(kernelname);
    }
    else{
      kernel_count = 0;
      getOperation().walk([&](ADORA::KernelOp kn){
        std::string kernelname = getOperation().getSymName().str() + 
          "_" + std::to_string(kernel_count++);
        kn.setKernelName(kernelname); 
      });
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::ADORA::createExtractAffineForToKernelPass() {
  return std::make_unique<AffineForKernelCaptor>();
}


