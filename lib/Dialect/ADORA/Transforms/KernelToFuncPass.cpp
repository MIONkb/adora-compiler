//===- KernelToModulePass.cpp - Convert a kernel to a Module file which will be optimized -----------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/AffineExprVisitor.h"
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/RegionUtils.h"

#include <iostream>
// #include <fstream>
#include <filesystem>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

#include "RAAA/Dialect/ADORA/IR/ADORA.h"
#include "RAAA/Dialect/ADORA/Transforms/Passes.h"
#include "./PassDetail.h"

using namespace llvm; // for llvm.errs()
using namespace llvm::detail;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;

//===----------------------------------------------------------------------===//
// KERNELToFunc
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "adora-kernel-to-function"
namespace
{

  // A pass that traverses Kernels in the function and extracts them to
  // individual Func/Module.
  struct KernelToFuncPass : public ExtractKernelToFuncBase<KernelToFuncPass>
  {
  public:
    void runOnOperation() override;

    //// Following functions are for pass option "kernel-explicit-datablock-trans"
    // getMemrefHeadAndFootprint(Kernel);
    // void ExplicitKernelDataBLockLoadStore(ADORA::KernelOp Kernel);
    void EliminateOuterLoopAffineTrans(ADORA::KernelOp Kernel);
    AffineExpr getConstPartofAffineExpr(AffineExpr &expr);
  };
} // namespace

static void EraseFuncAndReserveDeclaration(func::FuncOp func){
  OpBuilder b(func);
  FunctionType funcTy = func.getFunctionType();
  func::FuncOp declaration = b.create<func::FuncOp>(func.getLoc(), func.getSymName(), funcTy);
  declaration.setPrivate();

  func.erase();
}


void KernelToFuncPass::runOnOperation()
{
  SymbolTable symbolTable(getOperation());
  // bool modified = false;
  unsigned cnt = 0;
  std::map<unsigned, ADORA::KernelOp> cnt_to_KernelOP;

  for (auto FuncOp : getOperation().getOps<func::FuncOp>())
  {

    auto FuncWalkResult = FuncOp.walk([&](ADORA::KernelOp Kernel)
    {
      OpBuilder builder(Kernel);
      // Insert just after the function.
      // Block::iterator insertPt(Kernel.getOperation()->getNextNode());
      llvm::SetVector<Value> operands;
      std::string kernelFnName =\
            Twine(Kernel->getParentOfType<func::FuncOp>().getName()).concat("_kernel_"+std::to_string(cnt)).str();
      Kernel.setKernelName(kernelFnName);
      // Pull in instructions that can be sunk
      if (failed(sinkOperationsIntoKernelOp(Kernel)))
        return WalkResult::interrupt();
      
      // ///////////////
      // /// Generate explicit data block movement (load/store) for kernel to consume
      // ///////////////
      // if(ExplicitDataTrans==true){
      //   /// generate explicit data movement around Kernel{...}
      //   llvm::errs() << "[dubug] Before ExplicitKernelDataBLockLoadStore: \n";Kernel.dump();
      //   mlir::ADORA::ExplicitKernelDataBLockLoadStore(Kernel);
      //   llvm::errs() << "[dubug] After ExplicitKernelDataBLockLoadStore: \n";Kernel.dump();

        //   /// Eliminate the affine transformation of the upper/lower bound 
        //   /// of most-out loop in Kernel{...}
        //   mlir::ADORA::EliminateOuterLoopAffineTrans(Kernel);

        //   llvm::errs() << "[dubug] After ExplicitKernelDataBLockLoadStore, Kernel: \n";Kernel.dump();
        //   /// Remove unused arguments of Kernel's region
        //   // Kernel.walk([&](Region *region){ removeUnusedRegionArgs(*region); });
        // }

      func::FuncOp NewKernelFunc = GenKernelFunc(Kernel, operands);
      symbolTable.insert(NewKernelFunc);

      /// If option "kernel-gen-dir" is set
      if(KernelGenDir != ""){ 

        std::filesystem::create_directory(KernelGenDir+"/kernels");
        std::string KernelFilePath_str = KernelGenDir+"/kernels/"+ kernelFnName + ".mlir";
        std::error_code ec;
        llvm::raw_fd_ostream file(KernelFilePath_str, ec, sys::fs::FA_Write);
        if (ec) {
          llvm::errs() << "Error opening file: " << ec.message() << KernelFilePath_str << "\n";
          return WalkResult::advance();
        }
        LLVM_DEBUG(llvm::errs() << "Kernel:"  << kernelFnName << "\n";);

        // ModuleOp new_m = builder.create<ModuleOp>(NewKernelFunc.getLoc(), kernelFnName);
        // new_m.push_back(NewKernelFunc.clone());

        LLVM_DEBUG(NewKernelFunc.dump(););
        file << NewKernelFunc;
        // new_m.dump();
        // NewKernelFunc.print(llvm::errs());
        // file << new_m;
        // new_m.erase();

        /// Convert ADORA.Kernel{ ... } to func.call
        /// Kernel call function is substituded
        // builder.create<ADORA::KernelCallOp>(Kernel.getLoc(), NewKernelFunc, operands.getArrayRef());
        builder.create<func::CallOp>(Kernel.getLoc(), NewKernelFunc, operands.getArrayRef());
        LLVM_DEBUG(getOperation()->dump(););
        EraseFuncAndReserveDeclaration(NewKernelFunc);
        // NewKernelFunc.erase();
       LLVM_DEBUG(getOperation()->dump(););

      }

      else{
        /// Convert ADORA.Kernel{ ... } to func.call
        builder.create<ADORA::KernelCallOp>(Kernel.getLoc(), NewKernelFunc, operands.getArrayRef());
      }



      // Kernel->erase();
    
      cnt_to_KernelOP[cnt] = Kernel;
      cnt++;

      return WalkResult::advance();
    });

    
    if (FuncWalkResult.wasInterrupted())
      return signalPassFailure();
  }
  
  // assert(cnt == 1 && "There should be only 1 topFunc in IR Module.");

  unsigned kernel_num = cnt;
  for (cnt = 0; cnt < kernel_num; cnt++)
  {
    assert(cnt_to_KernelOP[cnt] && "Counter and Kernel did not match!");
    cnt_to_KernelOP[cnt].erase();
  }
  // std::cout << "[debug] after erase:\n"; topFunc.dump();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::ADORA::createExtractKernelToFuncPass()
{
  return std::make_unique<KernelToFuncPass>();
}
