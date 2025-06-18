//===- AutoDesignSpaceExplore.cpp - design space explore for application frontend optimization -----------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/SourceMgr.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GraphWriter.h"

#include <numeric>
#include <iostream>
#include <filesystem>
#include <fstream>
// #include <regex>
#include <stack>

#include "RAAA/Dialect/ADORA/IR/ADORA.h"
#include "RAAA/Dialect/ADORA/Transforms/Passes.h"
#include "RAAA/Dialect/ADORA/Transforms/DSE.h"
#include "../../../DFG/inc/mlir_cdfg.h"
#include "RAAA/Misc/DFG.h"
#include "./PassDetail.h"


using namespace llvm; // for llvm.errs()
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;


/********** 
 * TODO: modify this
Steps to achieve the DSE process:

***/

namespace {

struct AutoDesignSpaceExplorer : public AutoDesignSpaceExploreBase<AutoDesignSpaceExplorer> {
  AutoDesignSpaceExplorer() = default;


  /* Function define */
  // ADORA::ForNode* findTargetLoopNode(SmallVector<ADORA::ForNode>& NodeVec, mlir::affine::AffineForOp forop);
  // void NestedGenTree(ADORA::ForNode*, SmallVector<ADORA::ForNode>&);
  // SmallVector<ADORA::ForNode> createAffineForTree(func::FuncOp topfunc);
  // SmallVector<unsigned> FindUnrollingFactors(ADORA::ForNode& Node); 
  // SmallVector<DesignPoint> ExpandTilingAndUnrollingFactors(ADORA::ForNode Node, SmallVector<DesignPoint> CurrentDesignSpace);
  SmallVector<DesignPoint> ConstructTilingUnrollSpace(SmallVector<ADORA::ForNode> ForNodes);
  // std::string GenDFGfromAffinewithCMD(std::string KernelsDir, std::string kernelFnName);
  // DFGInfo GetDFGinfo(std::string DFGPath);
  void runOnOperation() override; 

};
} // namespace

/// @brief A design point is a number sequence:
///        tilefactor(loop0),unrollfactor(loop0),tilefactor(loop1),unrollfactor(loop1).......
/// for example:
///       (1,1),(1,1),(1,3) represents a 3-level nested loop and the innermost loop is unrolled by 3
/// @param ForNodes All for loops
/// @return A SmallVector containing all design point
SmallVector<DesignPoint> AutoDesignSpaceExplorer::
            ConstructTilingUnrollSpace(SmallVector<ADORA::ForNode> ForNodes){
  SmallVector<DesignPoint> AllDesignSpace;
  DesignPoint point;
  unsigned point_num = 1;
  for (ADORA::ForNode Node : ForNodes) {
    AllDesignSpace = ExpandTilingAndUnrollingFactors(Node, AllDesignSpace);
    point_num *= Node.UnrollFactors.size();
  }

  /*** Print Design Space ***/
  llvm::errs() <<"//-----------  Design Space  ----------//\n";
  llvm::errs() <<"Loop-nested structure:\n";
  for (ADORA::ForNode Node : ForNodes) {
    if (!Node.HasParentFor())/// outermost loop
      Node.dumpTree();
  }

  llvm::errs() <<"Design Space:\n";
  for(DesignPoint point : AllDesignSpace){
    bool bracket_flag = 1;
    for(unsigned factor : point){
      if(bracket_flag) 
        llvm::errs() << "(";

      llvm::errs() << factor;

      if(bracket_flag) 
        llvm::errs() << ",";
      else 
        llvm::errs() << ")";

      bracket_flag = !bracket_flag;
    }
    llvm::errs() << "\n";
  }

  llvm::errs() <<"//-------------------------------------//\n";
  return AllDesignSpace;
}


/// @brief 
void AutoDesignSpaceExplorer::runOnOperation(){
  ModuleOp topmodule = getOperation();
  MLIRContext* context = topmodule.getContext();
  auto originmodule = topmodule.getOperation()->clone();
  unsigned func_cnt = 0;
  SmallVector<DesignPoint> AllDesignSpace;
  SmallVector<std::string> DesignSpaceFiles;
  for (auto func : topmodule.getOps<func::FuncOp>()) {
    SmallVector<ADORA::ForNode> ForNodes = createAffineForTree(func);

    /** construct design space **/
    for (ADORA::ForNode& Node : ForNodes) {
      /* tiling if perfect */
      /// TODO: Tiling
      // if (Node.IsThisLevelPerfect()){
      // }
      /* unrolling if innermost */
      if (Node.IsInnermost()){
        FindUnrollingFactors(Node);
      }
    }

    // construct design space through all possible unroll factors
    AllDesignSpace = ConstructTilingUnrollSpace(ForNodes);

    func_cnt++;
  }
  assert(func_cnt == 1 && "Can't handle subfunctions.");

  /** Make a directory to contain all design points **/
  // Get current path
  std::filesystem::path currentPath = std::filesystem::current_path();

  // make a new dir
  std::string folderName = "DesignSpace";
  std::filesystem::path DesignSpacefolderPath = currentPath / folderName;
  std::filesystem::create_directory(DesignSpacefolderPath);

  /** Traverse the whole design space **/
  for(DesignPoint point : AllDesignSpace){
    topmodule = cast<ModuleOp>(originmodule->clone());
    func_cnt = 0;
    for (auto func : topmodule.getOps<func::FuncOp>()) {
      SmallVector<ADORA::ForNode> ForNodes = createAffineForTree(func);
      std::string fileName = func.getName().str();
      for(unsigned node_cnt = 0; node_cnt < ForNodes.size(); node_cnt++){
        /// update filename 
        fileName += "_" + std::to_string(point[2 * node_cnt]) 
                  + "_" + std::to_string(point[2 * node_cnt+1]);
        // llvm::errs() <<"filename:" << fileName << "\n";
        /// Skip the tiling factor
        
        /// Apply unroling by this factor
        unsigned unrollfactor = point[2 * node_cnt + 1];
        ADORA::ForNode NodeToUnroll = ForNodes[node_cnt];
        if(unrollfactor > 1){
          if(failed(loopUnrollByFactor(NodeToUnroll.getForOp(), 
                unrollfactor, /*annotateFn=*/nullptr, /*cleanUpUnroll=*/false))){
            llvm::errs() << "Unroll failed!!!\n";
            return signalPassFailure();
          }
        }
        // llvm::errs() << "[debug] new for:\n";
        // NodeToUnroll.dumpForOp();
      }

      /// create a new mlir file
      std::string filePath = DesignSpacefolderPath.string() + "/" + fileName + ".mlir";
      llvm::errs() << "filePath: " << filePath << "\n";
      std::error_code ec;
      llvm::raw_fd_ostream outputFile(filePath, ec, sys::fs::FA_Write);
      if (ec) {
        llvm::errs() << "Error opening file: " << ec.message() << filePath << "\n";
        return signalPassFailure();
      }
      DesignSpaceFiles.push_back(fileName);

      topmodule.print(outputFile);
      llvm::errs() << "[Info] design point " << fileName << ":\n";
      topmodule.dump();

      func_cnt++;
    }
    assert(func_cnt == 1 && "Can't handle subfunctions.");
  }

  //////////////////////////
  /// Find the best unrolling factor for every kernel if CGRA ADG and 
  /// llvm DFG generator shared library(.so) is specified. 
  //////////////////////////
  if(CGRAadg == "notdefined"){
    llvm::errs() << "[Info]No cgra-adg specified. Generate design space only.\n";
    return;
  }
  if(llvmCDFGPass == "notdefined"){
    llvm::errs() << "[Info]No DFG generator shared library(.so) specified." 
                  << " Generate design space only.\n";
    return;
  }

  // Read target adg JSON file.
  std::string errorMessage;
  auto ADGFile = mlir::openInputFile(CGRAadg, &errorMessage);
  if (!ADGFile) {
    llvm::errs() << errorMessage << "\n";
    return signalPassFailure();
  }

  // Parse JSON file into memory.
  auto ADGjson = llvm::json::parse(ADGFile->getBuffer());
  if (!ADGjson) {
    llvm::errs() << "failed to parse the target cgra adg json file\n";
    return signalPassFailure();
  }
  auto ADGObj = ADGjson.get().getAsObject();
  if (!ADGObj) {
    llvm::errs() << "support an object in the target spec json file, found "
                  <<    "something else\n";
    return signalPassFailure();
  }

  // ADG adg;

  // unsigned SizeSpadBank = ADGObj->getInteger("iob_spad_bank_size").value_or(8192);
  ///TODO: Size Scratchpad 
  unsigned NumGPE = 0;
  unsigned NumIOB = 0;  
  auto instances = ADGObj->getArray("instances");   
  for(auto insJson : *instances){
    auto insJsonObj = insJson.getAsObject();
    auto insType = insJsonObj->getString("type").value();
    // llvm::errs() << "[info] type:" << insType << "\n";      
    if(insType == "GPE")
      NumGPE++;
    else if(insType == "IOB")
      NumIOB++;
  }

  for(auto DesignPointFile : DesignSpaceFiles){
    std::string FilePath = DesignSpacefolderPath.string() 
                                    + "/" + DesignPointFile + ".mlir";
    // llvm::errs() << "FilePath:" << FilePath << "\n";
    // llvm::StringRef FileName = DesignPointFile;
    // Set up the input file.
    std::string errorMessage;
    auto file = openInputFile(FilePath, &errorMessage);
    if (!file) {
      llvm::errs() << errorMessage << "\n";
      assert(0);
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> m = parseSourceFile<ModuleOp>(sourceMgr, context); 
    mlir::ModuleOp moduleop = m.get();
    SymbolTable symbolTable(moduleop.getOperation());
    // llvm::errs() << "module:" << moduleop << "\n";

    //////////////////
    /// Extract all affine-for kernel
    //////////////////
    unsigned cnt = 0;
    for (auto func : moduleop.getOps<func::FuncOp>()) {
      for (Operation &op : llvm::make_early_inc_range(func.getOps())) {
        if (auto forOp = dyn_cast<AffineForOp>(&op)) {
          (void)ADORA::SpecifiedAffineFortoKernel(forOp);
        }
      }

      func.walk([&](ADORA::KernelOp Kernel)
      {
        // Insert just after the function.
        // Block::iterator insertPt(Kernel.getOperation()->getNextNode());
        auto KernelWalkResult = Kernel.walk([&](ADORA::KernelOp op) 
        {
          llvm::SetVector<mlir::Value> operands;
          std::string kernelFnName = "kernel_"+std::to_string(cnt);
          Kernel.setKernelName(kernelFnName);
          // Pull in instructions that can be sunk
          if (failed(sinkOperationsIntoKernelOp(op)))
            return WalkResult::interrupt();
      
          // ///////////////
          // /// Generate explicit data block movement (load/store) for kernel to consume
          // ///////////////
          // if(ExplicitDataTrans==true){
          //   /// generate explicit data movement around Kernel{...}
          //   // llvm::errs() << "[dubug] Before ExplicitKernelDataBLockLoadStore: \n";topFunc.dump();
          //   ExplicitKernelDataBLockLoadStore(Kernel);
          //   // llvm::errs() << "[dubug] After ExplicitKernelDataBLockLoadStore: \n";topFunc.dump();
        
          //   /// Eliminate the affine transformation of the upper/lower bound 
          //   /// of most-out loop in Kernel{...}
          //   EliminateOuterLoopAffineTrans(Kernel);
          // }

          func::FuncOp NewKernelFunc = GenKernelFunc(op, operands);
          symbolTable.insert(NewKernelFunc);

          /// If option "kernel-gen-dir" is set
          OpBuilder builder(NewKernelFunc);
          std::string KernelsDir = DesignSpacefolderPath.string() 
                                      + "/"+ DesignPointFile + "_kernels";

          std::filesystem::create_directory(KernelsDir);
          std::string KernelFilePath_str = KernelsDir+"/"+ kernelFnName + ".mlir";
          std::error_code ec;
          llvm::raw_fd_ostream file(KernelFilePath_str, ec, sys::fs::FA_Write);
          if (ec) {
            llvm::errs() << "Error opening file: " << ec.message() << KernelFilePath_str << "\n";
            return WalkResult::advance();
          }
          llvm::errs() << "Kernel:"  << kernelFnName << "\n";
          NewKernelFunc.dump();
          // NewKernelFunc.print(llvm::errs());
          NewKernelFunc.print(file);
          file.close();
          NewKernelFunc.erase();
          
          /// Generate DFG with linux system command
          std::string DFGPath = GenDFGfromAffinewithCMD(KernelsDir, kernelFnName, llvmCDFGPass);
          if(DFGPath.empty()){
            llvm::errs() << "[Error] Generate DFG failed!";
            return WalkResult::interrupt();
          }

          /// Read in DFG from dot
          ADORA::DFGInfo dfginfo = GetDFGinfo(DFGPath);          

          /// Convert ADORA.Kernel{ ... } to func.call
          // OpBuilder builder(op);
          // builder.create<ADORA::KernelCallOp>(op.getLoc(), NewKernelFunc, operands.getArrayRef());
          return WalkResult::advance();
        });
        if (KernelWalkResult.wasInterrupted())
          return signalPassFailure();
        llvm::errs() << "module:" << moduleop << "\n";

        //////////////////
        /// Run lowering with linux system command
        /// TODO: use nested pass manager??
        //////////////////
        // mlir::PassManager nestedPM(&getContext());
        // nestedPM.addPass(std::make_unique<MyFirstPass>());
        // nestedPM.addPass(std::make_unique<MySecondPass>());

        // if (failed(nestedPM.run(getModule()))) {
        //   llvm::errs() << "Nested pipeline failed to run!\n";
        // }        

        // cnt_to_KernelOP[cnt] = Kernel;
        cnt++;
      });

    }

    llvm::errs() << "module:" << moduleop << "\n";
  }
}


std::unique_ptr<OperationPass<ModuleOp>> 
        mlir::ADORA::createAutoDesignSpaceExplorePass(/*std::string dseTargetSpec*/) {
  return std::make_unique<AutoDesignSpaceExplorer>(/*dseTargetSpec*/);
}
