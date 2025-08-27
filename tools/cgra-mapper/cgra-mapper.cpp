#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Debug/CLOptionsSetup.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/FileUtilities.h" 
#include "llvm/Support/MemoryBuffer.h"

#include "../../lib/DFG/inc/mlir_cdfg.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
// #include "ADORA/Dialect/ADORA/Transforms/Passes.h"
// #include "ADORA/Dialect/ADORA/Lowering/LowerPasses.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Misc/Passes.h"
#include "ADORA/Misc/DFG.h"

#include <iostream>
#include <set>
#include <cstdlib>
#include <ctime>
#include <regex>
#include <sstream>
#include <thread>
#include <mutex>
#include <getopt.h>

#include "op/operations.h"
#include "ir/adg_ir.h"
#include "ir/dfg_ir.h"
#include "mapper/mapper_sa.h"
#include "spdlog/spdlog.h"
#include "spdlog/cfg/argv.h"
#include "emit/EmitCGRACall.h"
#include "emit/EmitPytest.h"
#include "tensorop/TensorOp.h"

// #include "mlir/Dialect/Arith/Transforms/Passes.h"
// #include "mlir/Dialect/Func/Transforms/Passes.h"

// Defined in the test directory, no public header.
namespace mlir {
} // namespace mlir

using namespace llvm;
using namespace mlir;

// static int kernel_cnt = 0;

int main(int argc, char **argv) {
  // mlir::registerAllDialects();
  // mlir::registerAllPasses();

  spdlog::cfg::helpers::load_levels("true");
  mlir::DialectRegistry registry;

  //===--------------------------------------------------------------------===//
  // Register mlir dialects and passes
  //===--------------------------------------------------------------------===//
  // Add the following to selectively include the necessary dialects. You only
  // need to register dialects that will be *parsed* by the tool, not the one
  // generated
  // clang-format off
  registry.insert<mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect,
                  mlir::scf::SCFDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::vector::VectorDialect,
                  mlir::arith::ArithDialect,
                  mlir::affine::AffineDialect,
                  mlir::DLTIDialect,
                  mlir::ml_program::MLProgramDialect,
                  mlir::tensor::TensorDialect,
                  mlir::bufferization::BufferizationDialect>();

  // Dialects
  registry.insert<mlir::ADORA::ADORADialect,
                  mlir::ADORA::ADORATensor::ADORATensorDialect>();
  // return failed(
  //     mlir::MlirOptMain(argc, argv, "Fail\n", registry)
  // );

  //===--------------------------------------------------------------------===//
  // Similar to MlirOptMian() in MlirOptMain.cpp
  //===--------------------------------------------------------------------===//

  /// User args
  /// A good example to use cgra-mapper is:
  ///  
  static cl::opt<std::string> inputFilename(
    cl::Positional, 
    cl::desc("<input file>"), 
    cl::init("-"));

  // static cl::opt<bool> dumpCallFunc(
  //   "dump-call-func",
  //   cl::Optional, 
  //   cl::desc("dump call function of CGRA (default)"), 
  //   cl::init(false));

  static cl::opt<bool> dumpMappedViz(
    "dump-mapped-viz",
    cl::Optional, 
    cl::desc("dump-mapped-viz"), 
    cl::init(false));
  
  static cl::opt<bool> objOpt(
    "obj-opt",
    cl::Optional, 
    cl::desc("obj-opt"), 
    cl::init(true));

  static cl::opt<int> timeout_ms(
    "timeout",
    cl::Optional, 
    cl::desc("timeout(ms)"), 
    cl::value_desc("int"), 
    cl::init(360000));

  static cl::opt<int> max_iters(
    "max-iters",
    cl::Optional, 
    cl::desc("max-iters"), 
    cl::value_desc("int"), 
    cl::init(2000));
    
  static cl::opt<std::string> adg_fn(
    "adg",
    cl::Required, 
    cl::desc("adg file"), 
    cl::value_desc("adg filename"), 
    cl::init("-"));

  static cl::opt<std::string> op_fn(
    "op-file",
    cl::Required, 
    cl::desc("op file"), 
    cl::value_desc("op filename"), 
    cl::init("-"));

  static cl::opt<std::string> emit_type(
    "output-type",
    cl::Required, 
    cl::desc("emit the execution file type: c(defualt), pytest"), 
    cl::value_desc("c or pytest"), 
    cl::init("c"));

  static cl::opt<std::string> outputFilename(
    "output", 
    cl::Optional, 
    cl::desc("Output filename"),
    cl::value_desc("filename"),
    cl::init("-"));
  
  // static cl::opt<int> nthreads(
  //   "j", 
  //   cl::Optional, 
  //   cl::desc("Allow N mapping jobs at once(default to be 1)"),
  //   cl::value_desc("[N]"),
  //   cl::init(1));

  InitLLVM y(argc, argv);

  MlirOptMainConfig::registerCLOptions(registry);
  // registerAsmPrinterCLOptions();
  // registerMLIRContextCLOptions();
  // registerPassManagerCLOptions();
  // tracing::DebugCounter::registerCLOptions();

  // Build the list of dialects as a header for the --help message.
  std::string helpHeader = "\nAvailable Dialects: ";
  {
    llvm::raw_string_ostream os(helpHeader);
    interleaveComma(registry.getDialectNames(), os,
                    [&](auto name) { os << name; });
  }
  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, helpHeader);
  MlirOptMainConfig config = MlirOptMainConfig::createFromCLOptions();



  // When reading from stdin and the input is a tty, it is often a user mistake
  // and the process "appears to be stuck". Print a message to let the user know
  // about it!
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  context.getOrLoadDialect(mlir::ADORA::ADORADialect::getDialectNamespace());
  context.getOrLoadDialect(mlir::ADORA::ADORATensor::ADORATensorDialect::getDialectNamespace());

  if (inputFilename == "-" &&
      sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  std::string errorMessage;
  
  Twine t = (StringRef)inputFilename;
  // openInputFileImpl(t, errorMessage,
  //                          /*alignment=*/std::nullopt);
  // openInputFile((StringRef)inputFilename, &errorMessage);

  llvm::MemoryBuffer::getFileOrSTDIN(t);
  llvm::MemoryBuffer::getFileOrSTDIN(
      t, /*IsText=*/false, /*RequiresNullTerminator=*/true,
       /*alignment=*/std::nullopt);

  t.dump();


  /////////////////////////
  /// Parse input file
  /////////////////////////
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    assert(0);
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> m = parseSourceFile<ModuleOp>(sourceMgr, &context); 
  mlir::ModuleOp moduleop = m.get();
  // SymbolTable symbolTable(moduleop.getOperation());
  
  //////////////////////////////////////////
  /// Parse Operation file and ADG file
  //////////////////////////////////////////
  unsigned seed = time(0); // random seed using current time
  srand(seed);  // set random generator seed 
  std::cout << "Parse Operations: " << op_fn << std::endl;
  Operations::Instance(op_fn);
  // Operations::print();

  std::cout << "Parse ADG: " << adg_fn << std::endl;
  ADGIR adg_ir(adg_fn);
  ADG* adg = adg_ir.getADG();
  int numGpeNodes = adg->numGpeNodes();
  int numFuNodes = numGpeNodes + adg->numIobNodes();
  std::cout << "numGpeNodes: " << numGpeNodes << ", numFuNodes(GPE+IOB): "  << numFuNodes << std::endl;
  std::vector<float>storePEusage;
  std::vector<float>storeFUusage;
  std::vector<int>bestLatency;
  // adg->print();

  //////////////////////////////////////////
  /// Pre-set mapping
  //////////////////////////////////////////
  CGRACallEmitter CEmitter(moduleop);
  PytestEmitter PyEmitter(moduleop);
  std::vector<MapperSA*>mapper_Vec;
  std::vector<DFGIR*>DFGIR_Vec;

  std::vector<ADORA_TENSOR_MAPPER*>tensor_mapper_Vec;

  std::string GeneralOpNameFile_str;
  if (GeneralOpNameFile == nullptr) {
    std::cerr << "Environment variable \" GENERAL_OP_NAME_ENV \" is not set." << std::endl;
    GeneralOpNameFile_str = "/home/jhlou/CGRVOPT/cgra-opt/lib/DFG/Documents/GeneralOpName.txt";
    std::cerr << "Using \" GENERAL_OP_NAME_ENV \" = \"/home/jhlou/CGRVOPT/cgra-opt/lib/DFG/Documents/GeneralOpName.txt\"" << std::endl;
  }
  else
    GeneralOpNameFile_str = GeneralOpNameFile;

  /////////////////////////
  /// Map ADORA Tensor
  /////////////////////////
  MapAdoraTensorOp(&context, moduleop, tensor_mapper_Vec, adg, timeout_ms, max_iters, objOpt);
  // if(emit_type == "pytest"){
  //   MapAdoraTensorOp(tensor_mapper_Vec)
  // }
  // else{ /// default to be C
  //   CEmitter.preestablishPlacementConstraints(kernel, mapper);
  // }
  
  /////////////////////////
  /// Optimize module to make it suitable for emitting
  /////////////////////////
  /// Before emit C, simplify blockload and blockstore op and affineapply
  SimplifyBlockAccessOp(moduleop);
  ADORA::simplifyConstantAffineApplyOpsInRegion(moduleop.getBodyRegion());
  ADORA::simplifyAddAffineApplyOpsInRegionButOutOfKernel(moduleop.getBodyRegion());

  moduleop.dump();

  //////////////////////////////////////////
  /// Start mapping
  //////////////////////////////////////////
  
  /// Traverse through whole module to get a mapping result
  //// TODO: multithread mapping
  // SmallVector<ADORA::KernelOp> kernels;
  // moduleop.walk([&](ADORA::KernelOp kernel) {
  //   kernels.push_back(kernel);
  // });

  int kernel_cnt = 0;
  moduleop.walk([&](ADORA::KernelOp kernel) {
    MapperSA* mapper = new MapperSA(adg, timeout_ms, max_iters, objOpt);
    mapper_Vec.push_back(mapper);
    /// Generating DFG
    // std::string fileName = kernel.getKernelName();
  
    std::string kernelName = kernel.getKernelName();
    if(kernelName.empty()){
      kernelName = "kernel_" + std::to_string(kernel_cnt);
    }
    LLVMCDFG *CDFG = new LLVMCDFG(kernelName, GeneralOpNameFile_str);
    generateCDFGfromKernel(CDFG, kernel, /*verbose=*/true);
    // CDFG->CDFGtoDOT(CDFG->name_str()+"_CDFG.dot");

    /// DFG Mapping to CGRA architecture
    DFGIR* dfg_ir = new DFGIR(CDFG);
    DFGIR_Vec.push_back(dfg_ir);

    DFG* dfg = dfg_ir->getDFG();
    int numNodes = dfg->nodes().size();
    int numOpNodes = numNodes - dfg->ioNodes().size();
    std::cout << "numOpNodes: " << numOpNodes << ", numDfgNodes(Op+IO): "  << numNodes << std::endl;
    std::cout << "//============== Print DFG =================//" << std::endl;
    dfg->print();
    std::cout << "//============== End Print DFG =================//" << std::endl;
    // dfg->print();
    // map DFG to ADG
    mapper->setDFG(dfg);

    // some io nodes must be placed at some place
    if(emit_type == "pytest"){
      PyEmitter.preestablishPlacementConstraints(kernel, mapper);
    }
    else{ /// default to be C
      CEmitter.preestablishPlacementConstraints(kernel, mapper);
    }


    std::filesystem::create_directory(kernelName + "_map_result");
    CDFG->CDFGtoDOT(kernelName + "_map_result/before_map_" + CDFG->name_str() + "_CDFG.dot");
    bool succeed = mapper->execute(/*dumpCallFunc=*/false, /*dumpMappedViz*/true, /*resultDir=*/kernelName + "_map_result");
    // std::filesystem::create_directory("map_result");
    // CDFG->CDFGtoDOT("map_result/before_map_" + CDFG->name_str() + "_CDFG.dot");
    // bool succeed = mapper->execute(/*dumpCallFunc=*/false, /*dumpMappedViz*/true, /*resultDir=*/"map_result");
    if(succeed){
      // Mapping is successful, get all blockload and blockstore op and corresponding spad memory addresses.
      if(emit_type == "pytest"){
        PyEmitter.setMapResult(kernel, mapper);
        PyEmitter.DataBlockOperationsToSPADInfo(kernel, mapper);
        PyEmitter.GenerateCGRAConfig(kernel, mapper);
      }
      else{ /// default to be C
        CEmitter.setMapResult(kernel, mapper);
        CEmitter.DataBlockOperationsToSPADInfo(kernel, mapper);
        CEmitter.GenerateCGRAConfig(kernel, mapper);
      }
    }
    kernel_cnt++;
  });

  /// Emit module to a C source file
  moduleop.dump();

  if(emit_type == "pytest"){
    if(outputFilename == "-")
      PyEmitter.emitPytest(llvm::errs());
    else{
      std::error_code ec;
      llvm::raw_fd_ostream outputFile(outputFilename, ec, sys::fs::FA_Write);
      PyEmitter.emitPytest(outputFile);
    }
  }
  else{ /// default to be C
    if(outputFilename == "-")
      CEmitter.emitCGRACallFunction(llvm::errs());
    else{
      std::error_code ec;
      llvm::raw_fd_ostream outputFile(outputFilename, ec, sys::fs::FA_Write);
      CEmitter.emitCGRACallFunction(outputFile);
    }
  }

  moduleop.dump();

  /// free
  for(auto mapper: mapper_Vec)
    delete mapper;
  for(auto ir: DFGIR_Vec)
    delete ir;
  for(auto mapper: tensor_mapper_Vec)
    delete mapper;

  return 0; 
}
