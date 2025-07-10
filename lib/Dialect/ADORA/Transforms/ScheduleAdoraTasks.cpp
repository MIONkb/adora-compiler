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
#include "RAAA/Dialect/ADORA/Transforms/TaskGraph/TaskGraph.h"
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
  bool BlockContainsKernelOp(mlir::Block* b);

  void ScheduleADORATasksInFunction(func::FuncOp func);

  void runOnOperation() override;
};
/// @brief 
/// @param b block to check whether contains a kernelop. Kernel op nested in for op is skipped.
/// @return 
bool ScheduleADORATasksPass::BlockContainsKernelOp(mlir::Block* b){
  for(auto op : b->getOps<ADORA::KernelOp>()){
    if(isa<ADORA::KernelOp>(op)){
      return true;
    }
  }

  return false;
}

void generateTaskGraphFromBlock(TaskGraph* graph, mlir::Block* block){
  graph->setParentOp(block->getParentOp());
  /// validloads : data block which has already been loaded to on-chip memory
  std::map<ADORA::LocalMemAllocOp, LocalAllocNode*> validallocs; 

  /// validloads : data block which has already been loaded to on-chip memory
  std::map<ADORA::DataBlockLoadOp, BlockLoadNode*> validloads; 

  /// dirtystores : data block which has not been written back to main memory
  std::map<ADORA::DataBlockStoreOp, BlockStoreNode*> dirtystores;
  
  for(auto _it = block->begin(); _it != block->end(); _it++){
    mlir::Operation* op = &(*_it);
    // op->dump();
    if(isa<ADORA::KernelOp>(op)){
      ADORA::KernelOp kernelop = dyn_cast<ADORA::KernelOp>(op);
      KernelNode* kernelnode = new KernelNode(kernelop);
      graph->AddNodeAndAnalyzeDefaultDependency(kernelnode);
    }
    else if(isa<ADORA::DataBlockLoadOp>(op)){
      ADORA::DataBlockLoadOp blockloadop = dyn_cast<ADORA::DataBlockLoadOp>(op);
      BlockLoadNode* blockloadnode = new BlockLoadNode(blockloadop);
      graph->AddNodeAndAnalyzeDefaultDependency(blockloadnode);
      validloads[blockloadop] = blockloadnode;

      /// handle load-after-store dependency here
      for(auto& pair : dirtystores){
        ADORA::DataBlockStoreOp blockstoreop = pair.first;
        BlockStoreNode* blockstorenode = pair.second;
        if(checkDependencyBetweenBlockStoreAndBlockLoad(blockstoreop, blockloadop)){
          addConnectionBetweenTwoNode(blockstorenode, blockloadnode, /*dep=*/depType::Depend);
        }
      }

      /// handle load-after-load dependency here

    }
    else if(isa<ADORA::LocalMemAllocOp>(op)){
      ADORA::LocalMemAllocOp allocop = dyn_cast<ADORA::LocalMemAllocOp>(op);
      LocalAllocNode* allocnode = new LocalAllocNode(allocop);
      graph->AddNodeAndAnalyzeDefaultDependency(allocnode);
      validallocs[allocop] = allocnode;
    }
    else if(isa<ADORA::DataBlockStoreOp>(op)){
      ADORA::DataBlockStoreOp blockstoreop = dyn_cast<ADORA::DataBlockStoreOp>(op);
      BlockStoreNode* blockstorenode = new BlockStoreNode(blockstoreop);
      graph->AddNodeAndAnalyzeDefaultDependency(blockstorenode);
      dirtystores[blockstoreop] = blockstorenode;
    }
    else{
      ///// if some stored data is used, then this store node must be written back.
    }
  }

  // analyze other dependencies
  // /// first, load-after-store
  // for(auto _it = block->begin(); _it != block->end(); _it++){
  //   mlir::Operation* op = &(*_it);
  //   op->dump();
  //   if(isa<ADORA::DataBlockLoadOp>(op)){
  //     ADORA::DataBlockLoadOp blockloadop = dyn_cast<ADORA::DataBlockLoadOp>(op);
  //     BlockLoadNode* blockloadnode = visitedloads[blockloadop];
      
  //   }
  //   else if(isa<ADORA::DataBlockStoreOp>(op)){
  //     ADORA::DataBlockStoreOp blockstoreop = dyn_cast<ADORA::DataBlockStoreOp>(op);
  //     BlockStoreNode* blockstorenode = new BlockStoreNode(blockstoreop);
  //     graph->AddNodeAndAnalyzeDefaultDependency(blockstorenode);
  //     visitedstores[blockstoreop] = blockstorenode;
  //   }
  // }    

}

/// @brief 
/// @param graph 
void analyzeDependencyInGraph(TaskGraph* graph){
  //// firstly, 
}




/// @brief A wrapper
/// @param func 
void ScheduleADORATasksPass::ScheduleADORATasksInFunction(func::FuncOp func){
  //////////////
  /// 1st step: get all block that needs to be scanned
  //////////////
  SmallVector<mlir::Block*> blocks;
  for(auto _it = func.getBody().begin(); _it != func.getBody().end(); _it++){
    mlir::Block* block = &*(_it); 
    if (BlockContainsKernelOp(block)) {
      blocks.push_back(block);
    }
  }
  func.walk([&](AffineForOp forop){
    mlir::Block* _b =  forop.getBody();
    if(BlockContainsKernelOp(_b)){
      blocks.push_back(_b);
    }
  });


  //////////////
  /// 2nd step: generate task graph and generate dependencies
  //////////////
  /// skip this
  int idx = 0;
  for(auto block : blocks){
    TaskGraph* graph = new TaskGraph;
    generateTaskGraphFromBlock(graph, block);
    block->dump();
    graph->dumpGraph();

    std::string filename = "Block_" + std::to_string(idx) + "_TaskGraph_0.dot";
    graph->dumpGraphAsDot(filename);   

    //////////////
    /// 3rd step: analyze dependency of transfered data block
    ///   Following dependencies will be analyzed:
    ///   g
    //////////////
    analyzeDependencyInGraph(graph);

    //////////////
    /// 4th step: simplify redundant data block transfer op
    //////////////
    //// move out redundant blockload

    //// remove redundant blockstore-blockload
    graph->RemoveRedundantBlockStoreLoadPair();

    block->dump();
    filename = "Block_" + std::to_string(idx) + "_TaskGraph_1.dot";
    graph->dumpGraphAsDot(filename);   
    
    idx++;
  }
}

void ScheduleADORATasksPass::runOnOperation()
{
  ScheduleADORATasksInFunction(getOperation());

  return;
}

} // namespace


std::unique_ptr<OperationPass<func::FuncOp>> 
  mlir::ADORA::createScheduleADORATasksPass()
{
  return std::make_unique<ScheduleADORATasksPass>();
}