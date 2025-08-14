//===- KernelToModulePass.cpp - Convert a kernel to a Module file which will be optimized -----------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Affine/Analysis/Utils.h"
// #include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Support/LLVM.h"
// #include "mlir/Support/FileUtilities.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/RegionUtils.h"
// #include "mlir/Transforms/RegionUtils.h"
// #include "mlir/Transforms/DialectConversion.h"

// #include <iostream>
// #include <fstream>
// #include <filesystem>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Debug.h"

#include "RAAA/Dialect/ADORA/IR/ADORA.h"
#include "RAAA/Dialect/ADORA/Transforms/Passes.h"
#include "RAAA/Dialect/ADORA/Transforms/DSE.h"
#include "RAAA/Dialect/ADORA/Transforms/SimplifyLoadStore.h"
#include "./PassDetail.h"

using namespace llvm; // for llvm.errs()
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;

#define DEBUG_TYPE "adora-simplify-loadstore"
//===----------------------------------------------------------------------===//
// SimplifyLoadStoreInLoopNest
//===----------------------------------------------------------------------===//
/** 
 * In this pass, we simplify the load and store operations with two methods:
 * First, hoist loop store operations to outer level, from innermost to outermost level.
 * Second, remove redundant store-load.
*/


namespace
{
  // enum class PositionRelationInLoop { SameLevel, LhsOuter, RhsOuter, NotInSameLoopNest}; 
    /// Relative position of 2 operations in loop nested
    ///  SameLevel : lhs and rhs are in same level
    ///  LhsOuter : lhs is in outer loop level
    ///  RhsOuter : rhs is in outer loop level
    ///  NotInSameLoopNest : rhs and lhs is not in the same loop-nest
  struct SimplifyLoadStoreInLoopNestPass : public SimplifyLoadStoreInLoopNestBase<SimplifyLoadStoreInLoopNestPass>
  {
  public:
    // static bool LoadStoreSameMemAddr(AffineLoadOp loadop, AffineStoreOp storeop); // Move to Utility.cpp
    // PositionRelationInLoop getPositionRelationship(Operation* lhs, Operation* rhs); 
    void runOnOperation() override;

  };
} // namespace

#define PASS_NAME "ADORA-simplify-loadstore"


bool TwoLoadsAccessSameMemAddr(AffineLoadOp load0, AffineLoadOp load1) {
  return TwoAccessSameMemAddr(load0, load1);
}
bool StoreLoadAccessSameMemAddr(AffineStoreOp store0, AffineLoadOp load1) {
  return TwoAccessSameMemAddr(store0, load1);
}
bool LoadStoreAccessSameMemAddr(AffineLoadOp load0, AffineStoreOp store1) {
  return TwoAccessSameMemAddr(load0, store1);
}

static bool ConsecutiveLoadsAccessSameMemAddr(AffineLoadOp load0, AffineLoadOp load1){
  return ConsecutiveAccessSameMemAddr<AffineLoadOp, AffineLoadOp>(load0, load1);
}
static bool ConsecutiveStoreLoadAccessSameMemAddr(AffineStoreOp store0, AffineLoadOp load1){
  return ConsecutiveAccessSameMemAddr<AffineStoreOp, AffineLoadOp>(store0, load1);
}
static bool ConsecutiveStoresAccessSameMemAddr(AffineStoreOp store0, AffineStoreOp store1){
  return ConsecutiveAccessSameMemAddr<AffineStoreOp, AffineStoreOp>(store0, store1);
}

/// Remove Redundant Loads
template <typename RegionOpT>
static bool RemoveRedundantLoads(RegionOpT RegionOp){
  LLVM_DEBUG(llvm::errs() << "Before RemoveRedundantLoads:\n");
  LLVM_DEBUG(RegionOp.dump());

  //// remove Consecutive Loads which access the same mem Addr
  bool NoChange = false;
  while(!NoChange){ // Keep walking the func until no change occurs in this func
    NoChange = true;

    for(AffineLoadOp load : RegionOp.template getOps<AffineLoadOp>()){
      for(AffineLoadOp otherload : RegionOp.template getOps<AffineLoadOp>()){
        if(load == otherload){ continue; }
        else if(ConsecutiveLoadsAccessSameMemAddr(load, otherload)){
          otherload.getOperation()->replaceAllUsesWith(load);
          otherload->erase();
          NoChange = false;
          break;
        }
      }

      if(!NoChange){ break; }
    }
  }

  LLVM_DEBUG(llvm::errs() << "After RemoveRedundantLoads:\n");
  LLVM_DEBUG(RegionOp.dump());

  //// remove Consecutive store-load which access the same mem Addr
  NoChange = false;
  while(!NoChange){ // Keep walking the func until no change occurs in this func
    NoChange = true;

    for(AffineStoreOp store : RegionOp.template getOps<AffineStoreOp>()){
      for(AffineLoadOp load : RegionOp.template getOps<AffineLoadOp>()){
        if(store == load){ continue; }
        else if(ConsecutiveStoreLoadAccessSameMemAddr(store, load)){
          load.getOperation()->replaceAllUsesWith(store.getValue().getDefiningOp());
          load->erase();
          NoChange = false;
          break;
        }
      }
      
      if(!NoChange){ break; }
    }
  }


  LLVM_DEBUG(llvm::errs() << "After RemoveRedundantstore-load:\n");
  LLVM_DEBUG(RegionOp.dump());

  return true;
}


/// Remove Redundant Loads
template <typename RegionOpT>
static bool RemoveRedundantStores(RegionOpT RegionOp){
  LLVM_DEBUG(llvm::errs() << "Before RemoveRedundantStores:\n");
  LLVM_DEBUG(RegionOp.dump());

  //// remove Consecutive Loads which access the same mem Addr
  bool NoChange = false;
  while(!NoChange){ // Keep walking the func until no change occurs in this func
    NoChange = true;

    for(AffineStoreOp store : RegionOp.template getOps<AffineStoreOp>()){
      for(AffineStoreOp otherstore : RegionOp.template getOps<AffineStoreOp>()){
        if(store == otherstore){ continue; }
        else if(ConsecutiveStoresAccessSameMemAddr(store, otherstore)){
          // otherstore.getOperation()->replaceAllUsesWith(store);
          store->erase();
          NoChange = false;
          break;
        }
      }

      if(!NoChange){ break; }
    }
  }

  LLVM_DEBUG(llvm::errs() << "After RemoveRedundantStores:\n");
  LLVM_DEBUG(RegionOp.dump());

  return true;
}

void SimplifyLoadStoreOpsInFunc(func::FuncOp f){
  SmallVector<ADORA::KernelOp> kernels;
  f.walk([&](ADORA::KernelOp kernel){
    kernels.push_back(kernel);
  });

  for(ADORA::KernelOp kernel: kernels){
    SmallVector<ADORA::ForNode> ForNodes = createAffineForTreeInsideKernel(kernel);
    for(auto n : ForNodes){
      n.dumpForOp();
      mlir::affine::AffineForOp forop = n.getForOp();
      RemoveRedundantLoads(forop);
      HoistLoadStoreOpsInOp(forop);
    }
  }

  for(ADORA::KernelOp kernel: kernels){
    SmallVector<ADORA::ForNode> ForNodes = createAffineForTreeInsideKernel(kernel);
    for(auto n : ForNodes){
      // n.dumpForOp();
      mlir::affine::AffineForOp forop = n.getForOp();
      RemoveRedundantStores(forop);
    }
  }


  return ;
}



void SimplifyLoadStoreInLoopNestPass::runOnOperation()
{
  /////////
  ///  TODO:
  ///  1. if load and store in if-else region ?
  ///  2. what if the load-store pair become (1 load and 2 stores) or (2 loads with 1 store) ?
  /////////
  func::FuncOp Func = getOperation();

  SimplifyLoadStoreOpsInFunc(Func);

  LLVM_DEBUG(Func.dump());
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::ADORA::createSimplifyLoadStoreInLoopNestPass()
{
  return std::make_unique<SimplifyLoadStoreInLoopNestPass>();
}
