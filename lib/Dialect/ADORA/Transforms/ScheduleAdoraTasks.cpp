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

      // /// handle load-after-load dependency here
      // for(auto& pair : validloads){
      //   ADORA::DataBlockLoadOp visitedblockloadop = pair.first;
      //   BlockLoadNode* blockloadnode = pair.second;
      //   if(SameBlockLoad(visitedblockloadop, blockloadop)){
      //     addConnectionBetweenTwoNode(blockstorenode, blockloadnode, /*dep=*/depType::Depend);
      //   }
      // }

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

////////////////////////////////////////////////////
//// rewrite task graph through dependency analysis
////////////////////////////////////////////////////
/// @brief Removes redundant pairs of BlockStoreNode and BlockLoadNode in the task graph.
/// 
/// This function iterates through the nodes in the task graph and identifies pairs
/// of BlockLoadNode and BlockStoreNode that access the same memory block.
/// If such pairs are found, it connects the store node's kernel to the load node's
/// kernel, replaces the load node with the source node of the store, and schedules
/// the load node for deletion to optimize memory access and reduce redundancy.
void RemoveRedundantBlockStoreLoadPair(TaskGraph* graph){
  std::vector<TaskNode*> to_delete;
  std::vector<TaskNode*> nodes = graph->getAllNodes();
  for (TaskNode* node : nodes) {
    // dumpNode(node);
    if(isa<BlockLoadNode>(node)){
      /// Check each block store input to determine whether they access the same memory space.
      for(auto innode : node->getInNodes()){
        if(isa<BlockStoreNode>(innode)){
          BlockStoreNode* storenode = dyn_cast<BlockStoreNode>(innode);
          ADORA::DataBlockStoreOp store = storenode->getDataBlockStoreOp();
          BlockLoadNode* loadnode = dyn_cast<BlockLoadNode>(node);
          ADORA::DataBlockLoadOp load = loadnode->getDataBlockLoadOp();
          
          if(store.getTargetMemref() == load.getOriginalMemref()
            && AccessSameDataBlock(store, load)){
            mlir::Operation* source = GetTheSourceOperationOfBlockStore(store);
            TaskNode* sourcenode = graph->getNode(source);
            assert(sourcenode != nullptr);

            //// connect storenode's kernel to loadnode's kernel
            KernelNode* sourcekernel = storenode->getKernelNode();
            for(auto sinkkernel : loadnode->getKernelNodes()){
              addConnectionBetweenTwoNode(sourcekernel, sinkkernel, /*dep=*/depType::Depend);
            }

            //// replace loadnode with sourcenode
            loadnode->ReplaceAllUsesWith(sourcenode);

            //// TODO: remove store? don't.
            to_delete.push_back(dyn_cast<TaskNode>(loadnode));
          }
        }
      }
    }
  }
  for(auto node : to_delete){
    graph->DeleteNodeOperation(node);
  }
}

void RemoveRedundantBlockLoads(TaskGraph* graph){
  std::vector<TaskNode*> to_delete;
  // std::vector<TaskNode*> nodes = graph->getAllNodes();
  // std::unordered_map<mlir::Operation*, BlockLoadNode*> load_map; // Maps original loads to their corresponding BlockLoadNode

  // for (TaskNode* node : nodes) {
  //   if (isa<BlockLoadNode>(node)) {
  //     BlockLoadNode* loadnode = dyn_cast<BlockLoadNode>(node);
  //     ADORA::DataBlockLoadOp load = loadnode->getDataBlockLoadOp();

  //     // Check if this load already exists in the map
  //     auto it = load_map.find(load.getOriginalMemref());
  //     if (it != load_map.end()) {
  //       // If a previous load node exists, replace the current load node with it
  //       BlockLoadNode* existing_loadnode = it->second;

  //       // Connect the existing load node's kernel to any dependent kernels
  //       KernelNode* existing_kernel = existing_loadnode->getKernelNode();
  //       for (auto sinkkernel : loadnode->getKernelNodes()) {
  //         addConnectionBetweenTwoNode(existing_kernel, sinkkernel, /*dep=*/depType::Depend);
  //       }

  //       // Replace the current load node with the existing one
  //       loadnode->ReplaceAllUsesWith(existing_loadnode);

  //       // Mark the current load node for deletion
  //       to_delete.push_back(dyn_cast<TaskNode>(loadnode));
  //     } else {
  //       // If no existing load node, add this one to the map
  //       load_map[load.getOriginalMemref()] = loadnode;
  //     }
  //   }
  // }

  // // Delete redundant load nodes
  // for (auto node : to_delete) {
  //   graph->DeleteNodeOperation(node);
  // }
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
    RemoveRedundantBlockStoreLoadPair(graph);
    //// remove redundant blockload-blockload
    RemoveRedundantBlockLoads(graph);

    //////////////
    /// 5th step: fix id of data transfer 
    //////////////


    block->dump();
    filename = "Block_" + std::to_string(idx) + "_TaskGraph_1.dot";
    graph->dumpGraphAsDot(filename);   
    
    idx++;
  }

  func.dump();
  ResetIndexOfBlockAccessOpInFunc(func);
  func.dump();
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