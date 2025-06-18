//===--------------------------------------------------------------------------------------------------===//
//===- ScheduleADORATasks.cpp - Schedule ADORA CGRA tasks -----------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Builders.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Parser/Parser.h"
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>
// #include <fstream>
// #include <filesystem>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"

#include "RAAA/Dialect/ADORA/IR/ADORA.h"
#include "RAAA/Dialect/ADORA/Transforms/Passes.h"
#include "RAAA/Dialect/ADORA/Transforms/DependencyAnalysis.h"
#include "./PassDetail.h"

using namespace llvm; // for llvm.errs()
using namespace llvm::detail;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;
//===----------------------------------------------------------------------===//
// AdjustKernelMemoryFootprint to meet cachesize
//===----------------------------------------------------------------------===//

#define PASS_NAME   "adora-schedule-cgra-tasks"
#define DEBUG_TYPE  "adora-schedule-cgra-tasks"

namespace
{
struct ScheduleADORATasksPass : 
  public ScheduleADORATasksBase<ScheduleADORATasksPass>
{
public:
  void runOnOperation() override;
};

void ScheduleADORATasksPass::runOnOperation()
{
  //////////////
  /// 1st step: generate task flow graph
  //////////////

  //////////////
  /// 2nd step: annotate info of kernel op
  //////////////

  //////////////
  /// 3rd step: analyze dependency of transfered data block
  ///   Following dependencies will be analyzed:
  ///   
  //////////////

  //////////////
  /// 4th step: simplify redundant data block transfer op
  //////////////

  return;
}

} // namespace


std::unique_ptr<OperationPass<ModuleOp>> 
  mlir::ADORA::createScheduleADORATasksPass()
{
  return std::make_unique<ScheduleADORATasksPass>();
}