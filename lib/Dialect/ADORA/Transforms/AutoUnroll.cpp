//===- AutoUnroll.cpp - Code to perform automated check between loop unroll, or unroll and jam ---------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/FileSystem.h"
#include <optional>
#include <filesystem>

#include "RAAA/Dialect/ADORA/IR/ADORA.h"
#include "RAAA/Dialect/ADORA/Transforms/DSE.h"
#include "RAAA/Dialect/ADORA/Transforms/Passes.h"
#include "RAAA/Misc/DFG.h"
#include "./PassDetail.h"

#define DEBUG_TYPE "adora-auto-unroll"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;

namespace {
  /// Automated unroll pass. Decide on whether to use unroll or unroll and jam
  /// with dependency analysis
  struct ADORAAutoUnroll
      : public ADORAAutoUnrollBase<ADORAAutoUnroll> {
    unsigned NumGPE = 0;
    unsigned NumIOB = 0; 
    explicit ADORAAutoUnroll() {
      if(CGRAadg == "notdefined" || CGRAadg == ""){
        LLVM_DEBUG(llvm::errs() << "CGRAadg not defined.\n");
        NumGPE = 32;
        NumIOB = 16;
      }
      else{
        NumGPE = getInstanceNumFromADG(CGRAadg, "GPE");
        NumIOB = getInstanceNumFromADG(CGRAadg, "IOB");
      }
      // if (unrollJamFactor)
      //   this->unrollJamFactor = *unrollJamFactor;
    }
    SmallVector<SmallVector<unsigned>> ConstructUnrollSpaceFromStrategy(SmallVector<ADORA::ForNode> ForNodes);
    LogicalResult chooseAndApplyUnrollStrategyWithDeps(ADORA::KernelOp kernel, mlir::ModuleOp& m);
    bool KernelIsInPerfectNestedLoop(ADORA::KernelOp kernel);
    void runOnOperation() override;
  };
} // namespace


std::unique_ptr<OperationPass<mlir::ModuleOp>>
    mlir::ADORA::createADORAAutoUnrollPass() {
  return std::make_unique<ADORAAutoUnroll>();
}

//////////
// Return true if operand "range" contains "value".
/////////
bool OperandRangeContainsValue(::mlir::Operation::operand_range range,  mlir::Value value){
  for(auto v : range){
    if(value == v)
      return true;
  }
  return false;
}

bool ADORAAutoUnroll::KernelIsInPerfectNestedLoop(ADORA::KernelOp kernel){
  mlir::Operation* parent = kernel.getOperation()->getParentOp();
  if(!isa_and_present<AffineForOp>(parent))
    return false;
  
  AffineForOp parentfor = dyn_cast<AffineForOp>(parent);

  // We already know that the block can't be empty.
  auto hasTwoElements = [](Block *block) {
    auto secondOpIt = std::next(block->begin());
    return secondOpIt != block->end() && &*secondOpIt == &block->back();
  };

  // parentForOp's body should be just this kernel and the terminator.
  if (!hasTwoElements(parentfor.getBody()))
    return false;

  return true;
}

/// @brief A design point is a number sequence:
///        tilefactor(loop0),unrollfactor(loop0),tilefactor(loop1),unrollfactor(loop1).......
/// for example:
///       (1,1),(1,1),(1,3) represents a 3-level nested loop and the innermost loop is unrolled by 3
/// @param ForNodes All for loops
/// @return A SmallVector containing all design point
SmallVector<SmallVector<unsigned>> ADORAAutoUnroll::ConstructUnrollSpaceFromStrategy(SmallVector<ADORA::ForNode> ForNodes){
  using namespace llvm; // for llvm.errs()
  SmallVector<SmallVector<unsigned>> AllDesignSpace;

  // unsigned point_num = 1;
  for (int i = 0; i < ForNodes.size(); i++) {
    ADORA::ForNode n = ForNodes[i];
    assert(ForNodes[i].IsOutOfKernel() || ForNodes[i].getUnrollStrategy() != UnrollStrategy::Undecided);

    if(ForNodes[i].getUnrollStrategy() == UnrollStrategy::Unroll){
      /// Find all unroll factor
      FindUnrollingFactors(ForNodes[i]); 

      UnrollDesignPoint point;

      /// For loops outer than this level, all factor will be 1.
      for(int j = 0; j < i; j++){
        point.push_back(ForNodes[j].getMaxUnrollFactor());
      }
      point.push_back(0); // point[i]
      /// For loops inner than this level, all factor will be tripcounts(fully unroll).
      for(int j = i + 1; j < ForNodes.size(); j++){
        point.push_back(1);
      }

      /// For this level loop,store all feasible factor.
      for(auto UF: ForNodes[i].UnrollFactors){
        point[i] = UF;
        if(findElement(AllDesignSpace, point) == -1){
          AllDesignSpace.push_back(point);
        }
      }
    }
    else if(ForNodes[i].getUnrollStrategy() == UnrollStrategy::Unroll_and_Jam){
      if(i == ForNodes.size() - 1) /// there is no loop level out of kernel
        break;
        
        
      assert(i < ForNodes.size() - 1); 
      ADORA::ForNode parentForNode = ForNodes[ForNodes.size() - 1];
      if(parentForNode.IsOutOfKernel()){
        /// Kernel is in one affine for op
        SmallVector<unsigned> parentForUnrollFactors = FindUnrollingFactors(parentForNode);
        
        UnrollDesignPoint point;
        /// For loops inner than this level, all factor will be tripcounts(fully unroll).
        for(int j = 0; j < i; j++){
          point.push_back(ForNodes[j].getMaxUnrollFactor());
        }

        /// For loops equal to or outer than this level, all factor will be 1.
        for(int j = i; j < ForNodes.size() - 1; j++){
          point.push_back(1);
        }

        /// For parent level loop,store all feasible factor.
        point.push_back(0);
        for(auto UF: parentForUnrollFactors){
          point[ForNodes.size() - 1] = UF;
          if(findElement(AllDesignSpace, point) == -1){
            AllDesignSpace.push_back(point);
          }
        }
      }
      // ForNodes[i].UnrollFactors.push_back(1);
      /// if one level can't be unrolled, then the outer level can't be unrolled either.
      break;
    }
    else if(ForNodes[i].getUnrollStrategy() == UnrollStrategy::CannotUnroll){
      ForNodes[i].UnrollFactors.push_back(1);
      /// If this level can't be unrolled, then out level can't be unrolled either.
      break;
    }
  }

  // /*** Print Design Space ***/
  llvm::errs() <<"//-----------  Design Space  ----------//\n";
  llvm::errs() <<"Loop-nested structure:\n";
  for (ADORA::ForNode Node : ForNodes) {
    if (!Node.HasParentFor())/// outermost loop
      Node.dumpTree();
  }

  llvm::errs() <<"Design Space:\n";
  for(UnrollDesignPoint point : AllDesignSpace){
    for (int i = 0; i < ForNodes.size(); i++) {
      llvm::errs() << point[i] << " ";
    }
    llvm::errs() << "\n";
  }

  llvm::errs() <<"//-------------------------------------//\n";
  return AllDesignSpace;
}

//////////
// A wrapper for KernelUnroll
/////////
LogicalResult ADORAAutoUnroll::
chooseAndApplyUnrollStrategyWithDeps(ADORA::KernelOp kernel, mlir::ModuleOp& m){
  /// make a new dir
  SmallVector<std::string> DesignSpaceFiles;
  std::filesystem::path currentPath = std::filesystem::current_path();
  std::string folderName = "DesignSpace";
  std::filesystem::path DesignSpacefolderPath = currentPath / folderName;
  std::filesystem::create_directory(DesignSpacefolderPath);
  int max_ALU = 0, max_LSU = 0;
  std::string final_FilePath="__";

  ///////////////
  /// For store op, check whether exists port conflict 
  ///////////////
  SmallVector<ADORA::ForNode> ForNodes = createAffineForTreeAroundKernel(kernel);
  // ForNodes[0].dumpTree();

  //// Decide the unroll strategy of each level(Unroll or Unroll-and-Jam)
  // SmallVector<UnrollStrategy> urStrategies;
  for(ADORA::ForNode& node: ForNodes){
    if(node.IsOutOfKernel())  /// Out of kernel
      continue;
    UnrollStrategy strategy;
    AffineForOp forop = node.getForOp();

    uint64_t StripCount = getConstantTripCount(forop).value_or(0);
    /// StripCount = 0, means the loop is not constant strip count, can't be unrolled
    if(StripCount == 0){
      strategy = UnrollStrategy::CannotUnroll;
    } 
    /// StripCount = 1, skip
    else if(StripCount == 1){
      strategy = UnrollStrategy::Unroll;
    }
    else{
      /// choose Unroll Strategy
      strategy = UnrollStrategy::Unroll;
      mlir::Value IterValue = forop.getSingleInductionVar().value();
      forop.walk([&](AffineStoreOp storeop){
        if(OperandRangeContainsValue(storeop.getIndices(), IterValue)){
          //// port conflict exists
          strategy = UnrollStrategy::Unroll_and_Jam;
          WalkResult::interrupt();
        }

        /// TODO: when to return CannotUnroll
      });
    }
    /// If kernel is not in a perfectly nested loop, cannot unroll.
    if(strategy == UnrollStrategy::Unroll_and_Jam && !KernelIsInPerfectNestedLoop(kernel)){
      strategy = UnrollStrategy::CannotUnroll;
    }
    node.setUnrollStrategy(strategy);
    // urStrategies.push_back(strategy);
    // urStrategies.push_back(UnrollStrategy::CannotUnroll);
  }

  ///// push parent for op 
  // affine::AffineForOp Parentfor = dyn_cast_or_null<affine::AffineForOp> 
  //                                 (kernel.getOperation()->getParentOp()); 
  // SmallVector<unsigned> parentForUnrollFactors;
  // if(Parentfor != nullptr){
  //   ADORA::ForNode parentForNode(Parentfor, -1/*Level: out of kernel*/);
  //   parentForUnrollFactors = FindUnrollingFactors(parentForNode);
  // }

  SmallVector<SmallVector<unsigned>> UnrollDesignSpace;
  UnrollDesignSpace = ConstructUnrollSpaceFromStrategy(ForNodes);
  //// Traverse the design space to unroll step by step from innermost to outermost.
  //// Unroll and jam if loops inside kernel cann't be unrolled 
  for(UnrollDesignPoint point : UnrollDesignSpace){
    ModuleOp topmodule = cast<ModuleOp>(m.getOperation()->clone());
    ADORA::KernelOp kernelur = getKernelFromCopiedModule(topmodule, kernel);
  
    SmallVector<ADORA::ForNode> ForNodesUR = createAffineForTreeAroundKernel(kernelur);
    std::string fileName;
    if(kernelur.hasKernelName())
      fileName = kernelur.getKernelName();
    else
      fileName = "Unroll";
    
    /// Unroll every level from innermost to outermost
    for(unsigned node_cnt = 0; node_cnt < ForNodesUR.size(); node_cnt++){
      /// update filename 
      fileName += "_" + std::to_string(point[node_cnt]);
      // llvm::errs() <<"filename:" << fileName << "\n";
      /// Skip the tiling factor
      
      /// Apply unrolling by this factor
      unsigned unrollfactor = point[node_cnt];
      ADORA::ForNode NodeToUnroll = ForNodesUR[node_cnt];
      NodeToUnroll.dumpForOp();

      /// if this level could be unrolled
      if(unrollfactor > 1 && !NodeToUnroll.IsOutOfKernel()){
        if(failed(ADORA::loopUnrollByFactor_opt(NodeToUnroll.getForOp(), 
              unrollfactor, /*annotateFn=nullptr,*/ /*cleanUpUnroll=*/false))){
            llvm::errs() << "Unroll failed!!!\n";
            signalPassFailure();
            return LogicalResult::failure();
        }
      }
      /// if this level is out of kernel and could be unroll-jam
      else if(unrollfactor > 1 && NodeToUnroll.IsOutOfKernel()){
        if(failed(ADORA::loopUnrollAndJamByFactor(NodeToUnroll.getForOp(), unrollfactor))){
            llvm::errs() << "Unroll failed!!!\n";
            signalPassFailure();
            return LogicalResult::failure();
        }
      }
      // llvm::errs() << "[debug] new for:\n";
      // NodeToUnroll.dumpForOp();
    }

    /// create a new mlir file
    std::string filePath = DesignSpacefolderPath.string() + "/" + fileName + ".mlir";
    // llvm::errs() << "filePath: " << filePath << "\n";
    std::error_code ec;
    llvm::raw_fd_ostream outputFile(filePath, ec, sys::fs::FA_Write);
    if (ec) {
      llvm::errs() << "Error opening file: " << ec.message() << filePath << "\n";
      signalPassFailure();
      return LogicalResult::failure();
    }
    DesignSpaceFiles.push_back(fileName);
  
    // llvm::errs() << "[Info] design point " << fileName << ":\n";
    topmodule.dump();
  
    simplifyConstantAffineApplyOpsInRegion(kernelur.getBody());
    simplifyLoadAndStoreOpsInRegion(kernelur.getBody());
    
    topmodule.print(outputFile);
    topmodule.dump();      
  
    /// Generating DFG
    std::string GeneralOpNameFile_str;
    if (GeneralOpNameFile == nullptr) {
      std::cerr << "Environment variable \" GENERAL_OP_NAME_ENV \" is not set." << std::endl;
      GeneralOpNameFile_str = "/home/jhlou/CGRVOPT/cgra-opt/lib/DFG/Documents/GeneralOpName.txt";
      std::cerr << "Using \" GENERAL_OP_NAME_ENV \" = \"/home/jhlou/CGRVOPT/cgra-opt/lib/DFG/Documents/GeneralOpName.txt\"" << std::endl;
    }
    else
      GeneralOpNameFile_str = GeneralOpNameFile;
    LLVMCDFG *CDFG = new LLVMCDFG(fileName, GeneralOpNameFile_str);
    generateCDFGfromKernel(CDFG, kernelur, /*verbose=*/true);
    CDFG->CDFGtoDOT(DesignSpacefolderPath.string() + "/" + CDFG->name_str()+"_CDFG_unroll.dot");
  
    ADORA::DFGInfo dfginfo = GetDFGinfo(CDFG);       
    if(dfginfo.Num_ALU <= NumGPE && dfginfo.Num_LSU <= NumIOB) 
    {
      if(dfginfo.Num_LSU > max_LSU || 
        (dfginfo.Num_ALU > max_ALU && dfginfo.Num_LSU == max_LSU))
      {
        final_FilePath = filePath;
        max_LSU = dfginfo.Num_LSU;
        max_ALU = dfginfo.Num_ALU;
      }
    }
    else{
      /// If the resource occupied by current dfg oversizes adg,
      /// we don't have to keep unrolling to get larger dfg.
      break;
    }

    // topmodule.erase();
  }/// End of traversing on every design point
  

  /// try to unroll and jam parent for of kernel
  // kernel.walk([&](AffineStoreOp storeop){
  //   // kernels.push_back(kernel);
  //   MemRefAccess storeAccess(storeop);
  //   unsigned depth = getNestingDepth(kernel.getTopAffineFopInKernel());
  
  if(final_FilePath != "__"){
    //// Apply the optimal unroll result
    std::string errorMessage;
    auto file = openInputFile(final_FilePath, &errorMessage);
    if (!file) {
      llvm::errs() << errorMessage << "\n";
      assert(0);
    }
    
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> final_m = parseSourceFile<ModuleOp>(sourceMgr, m.getContext()); 
    mlir::ModuleOp moduleop = final_m.get();
    SymbolTable symbolTable(moduleop.getOperation());
    
    // m.replace
    // for(int index = 0; index < m.getOps<func::FuncOp>().size(); index++){
    //   func::FuncOp oldfunc = *(m.getOps<func::FuncOp>()[index]);
    //   func::FuncOp newfunc = *(m.getOps<func::FuncOp>()[index]);
    //   newfunc.getOperation()->moveBefore(oldfunc);
    //   oldfunc.getOperation()->erase();
    // }
    func::FuncOp oldfunc = *(m.getOps<func::FuncOp>().begin());
    func::FuncOp newfunc = *(moduleop.getOps<func::FuncOp>().begin());
    newfunc.getOperation()->moveBefore(oldfunc);
    oldfunc.getOperation()->erase();
  }
 

  return LogicalResult::success();
}

void ADORAAutoUnroll::runOnOperation() {
  // if (getOperation().isExternal())
  //   return;
  auto m = getOperation();
  // func.dump();
  SmallVector<ADORA::KernelOp> kernels;
  m.walk([&](ADORA::KernelOp kernel){
    kernels.push_back(kernel);
  });

  for(ADORA::KernelOp kernel : kernels){
    LogicalResult r = chooseAndApplyUnrollStrategyWithDeps(kernel, m);
    m.dump();
  }

  for(auto func : m.getOps<func::FuncOp>()){
    ResetIndexOfBlockAccessOpInFunc(func);
  }
  m.dump();
}
