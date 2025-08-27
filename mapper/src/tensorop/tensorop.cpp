//===----------------------------------------------------------------------===//
//
// Copyright 2023-2024 The ADORA Authors.
//
//===----------------------------------------------------------------------===//

#include "tensorop/TensorOp.h"

namespace mlir{
namespace ADORA{

void MapAdoraTensorOp(mlir::ModuleOp moduleop, std::vector<ADORA_TENSOR_MAPPER*> mappers,
                    ADG* adg, int timeout_ms, int max_iters, bool objOpt){
  TensoDataflowGen engine;
  moduleop.walk([&](mlir::Operation* op) {
    if(engine.dispatchVisitor(op)){
      moduleop.dump();
    }
    // ADORA_TENSOR_MAPPER* mapper = new ADORA_TENSOR_MAPPER(adg, timeout_ms, max_iters, objOpt);
    // mappers.push_back(mapper);
    // /// Generating DFG
    // // std::string fileName = kernel.getKernelName();
  
    // std::string kernelName = kernel.getKernelName();
    // if(kernelName.empty()){
    //   kernelName = "kernel_" + std::to_string(kernel_cnt);
    // }
    // LLVMCDFG *CDFG = new LLVMCDFG(kernelName, GeneralOpNameFile_str);
    // generateCDFGfromKernel(CDFG, kernel, /*verbose=*/true);
    // // CDFG->CDFGtoDOT(CDFG->name_str()+"_CDFG.dot");

    // /// DFG Mapping to CGRA architecture
    // DFGIR* dfg_ir = new DFGIR(CDFG);
    // DFGIR_Vec.push_back(dfg_ir);

    // DFG* dfg = dfg_ir->getDFG();
    // int numNodes = dfg->nodes().size();
    // int numOpNodes = numNodes - dfg->ioNodes().size();
    // std::cout << "numOpNodes: " << numOpNodes << ", numDfgNodes(Op+IO): "  << numNodes << std::endl;
    // std::cout << "//============== Print DFG =================//" << std::endl;
    // dfg->print();
    // std::cout << "//============== End Print DFG =================//" << std::endl;
    // // dfg->print();
    // // map DFG to ADG
    // mapper->setDFG(dfg);

    // // some io nodes must be placed at some place
    // if(emit_type == "pytest"){
    //   PyEmitter.preestablishPlacementConstraints(kernel, mapper);
    // }
    // else{ /// default to be C
    //   CEmitter.preestablishPlacementConstraints(kernel, mapper);
    // }


    // std::filesystem::create_directory(kernelName + "_map_result");
    // CDFG->CDFGtoDOT(kernelName + "_map_result/before_map_" + CDFG->name_str() + "_CDFG.dot");
    // bool succeed = mapper->execute(/*dumpCallFunc=*/false, /*dumpMappedViz*/true, /*resultDir=*/kernelName + "_map_result");
    // // std::filesystem::create_directory("map_result");
    // // CDFG->CDFGtoDOT("map_result/before_map_" + CDFG->name_str() + "_CDFG.dot");
    // // bool succeed = mapper->execute(/*dumpCallFunc=*/false, /*dumpMappedViz*/true, /*resultDir=*/"map_result");
    // if(succeed){
    //   // Mapping is successful, get all blockload and blockstore op and corresponding spad memory addresses.
    //   if(emit_type == "pytest"){
    //     PyEmitter.setMapResult(kernel, mapper);
    //     PyEmitter.DataBlockOperationsToSPADInfo(kernel, mapper);
    //     PyEmitter.GenerateCGRAConfig(kernel, mapper);
    //   }
    //   else{ /// default to be C
    //     CEmitter.setMapResult(kernel, mapper);
    //     CEmitter.DataBlockOperationsToSPADInfo(kernel, mapper);
    //     CEmitter.GenerateCGRAConfig(kernel, mapper);
    //   }
    // }
    // kernel_cnt++;
  });

}

}
}