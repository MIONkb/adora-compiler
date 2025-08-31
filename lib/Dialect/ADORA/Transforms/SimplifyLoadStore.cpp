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

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Transforms/DSE.h"
#include "ADORA/Dialect/ADORA/Transforms/SimplifyLoadStore.h"
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

void SimplifyLoadStoreOpsInFunc(func::FuncOp f){
  SimplifyLoadStoreOpsInRegion(f.getBody());
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
