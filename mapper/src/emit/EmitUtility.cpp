#include "emit/Emit.h"
#include "mlir/Dialect/Affine/Utils.h"
//===----------------------------------------------------------------------===//
// Some tool functions
//===----------------------------------------------------------------------===//
std::vector<std::string> split_str_by_char(const std::string &s, const char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::string getEmitType(const mlir::Type valType){
   // Handle float types.
  if (valType.isa<Float32Type>())
    return std::string("float");
  else if (valType.isa<Float64Type>())
    return std::string("double");

  // Handle integer types.
  else if (valType.isa<IndexType>())
    return std::string("int");
  else if (auto intType = valType.dyn_cast<mlir::IntegerType>()) {
    if(intType.isInteger(16)){    
      return std::string("int16_t");
    }
    else if(intType.isInteger(32)){
      return std::string("int32_t");
    }
    else if(intType.isInteger(64)){ 
      return std::string("int64_t");
    }
    else if(intType.isUnsignedInteger(16)){    
      return std::string("uint16_t");
    }
    else if(intType.isUnsignedInteger(32)){
      return std::string("uint32_t");
    }
    else if(intType.isUnsignedInteger(64)){ 
      return std::string("uint64_t");
    }
  }

  return "-----------------error-----------------";
}

std::string getEmitType(const mlir::Value v){
  auto valType = v.getType();
  std::string type_str = getEmitType(valType);

  if(type_str == "-----------------error-----------------"){
    v.getDefiningOp()->emitError("Unsupported data type.");
    abort();
  }

  return type_str;
}

/// @brief A tool function to check whether a block access op is simplified
/// @param op can be datablockload or datablockstore op
/// @return
template <typename opT> bool IsSimplified(opT op){
  for(AffineExpr expr : op.getAffineMap().getResults()){
    switch (expr.getKind())
    {
    case AffineExprKind::DimId :
      continue;
      break;
    
    case AffineExprKind::Constant :
      continue;
      break;
  
    default:
      return false;
      break;
    }
  }
  return true;
}

/// @brief A function to retrieve the IOB IDs that can access a specified SPAD bank.
/// @param adg A pointer to an instance of the ADG class, which contains the mapping of IOBs to their connected SPAD banks.
/// @param bankId The identifier for the SPAD bank to check for accessibility.
/// @return A vector of IOB IDs that are connected to the specified SPAD bank.
std::vector<int> spadBankToIobs(ADG* adg, int bankId) {
    std::vector<int> accessibleIobs; // Vector to store IOBs that can access the specified bankId
    const auto& iobToBanksMap = adg->iobToSpadBanks(); // Get the mapping of all IOBs to their SPAD banks

    // Iterate over all IOBs
    for (const auto& entry : iobToBanksMap) {
      int iobId = entry.first; // Current IOB ID
      const std::vector<int>& connectedBanks = entry.second; // SPAD banks connected to the current IOB

      // Check if the current IOB is connected to the specified bankId
      if (std::find(connectedBanks.begin(), connectedBanks.end(), bankId) != connectedBanks.end()) {
        accessibleIobs.push_back(iobId); // If connected to bankId, add it to the result
      }
    }

    return accessibleIobs; // Return all IOB IDs that can access the specified bankId
}

/// @brief A function to simplify affine map of datablockload or datablockstore op
/// @param op 
void mlir::ADORA::SimplifyBlockAccessOp(mlir::ModuleOp m){
  m.walk([&](ADORA::DataBlockLoadOp blockload) {
    /// If block load op is simple, such as:
    /// %4 = ADORA.BlockLoad %arg2 [%arg3, %arg4, %arg5, %arg6]
    /// Then there is no need to simplify.
    if(IsSimplified(blockload))
      return WalkResult::advance(); ;

    mlir::OpBuilder b(blockload); 
    auto resultOperands =
        ::mlir::affine::expandAffineMap(b, blockload.getLoc(), blockload.getAffineMap(), blockload.getMapOperands());

    SmallVector<AffineExpr, 4> Exprs;
    for(int operandidx = 0; operandidx < (*resultOperands).size(); operandidx++){
      Value operand = (*resultOperands)[operandidx];
      // blockload.getOperation()->setOperand(operandidx + 1, operand);
      // operand.dump();
      AffineExpr Expr = b.getAffineDimExpr(operandidx);
      Exprs.push_back(Expr);
    }
    
    AffineMap map = AffineMap::get((*resultOperands).size(), /*symbolCount=*/0, Exprs, b.getContext());   /// stores corresponding AffineMap of above memIVs
    // map.dump();

    ADORA::DataBlockLoadOp newBlockLoad = b.create<ADORA::DataBlockLoadOp>\
                (blockload.getLoc(), blockload.getOriginalMemref(), map, *resultOperands, blockload.getResultType());

    newBlockLoad.setKernelName(blockload.getKernelName().str());
    newBlockLoad.setId(blockload.getId().str());

    blockload.getOperation()->replaceAllUsesWith(newBlockLoad);
    blockload.erase();
  });

  //// DataBlockStoreOp
  m.walk([&](ADORA::DataBlockStoreOp blockstore) {
    /// If block load op is simple, such as:
    /// %4 = ADORA.BlockLoad %arg2 [%arg3, %arg4, %arg5, %arg6]
    /// Then there is no need to simplify.
    if(IsSimplified(blockstore))
      return WalkResult::advance(); ;

    mlir::OpBuilder b(blockstore); 
    auto resultOperands =
        ::mlir::affine::expandAffineMap(b, blockstore.getLoc(), blockstore.getAffineMap(), blockstore.getMapOperands());

    SmallVector<AffineExpr, 4> Exprs;
    for(int operandidx = 0; operandidx < (*resultOperands).size(); operandidx++){
      Value operand = (*resultOperands)[operandidx];
      // blockload.getOperation()->setOperand(operandidx + 1, operand);
      // operand.dump();
      AffineExpr Expr = b.getAffineDimExpr(operandidx);
      Exprs.push_back(Expr);
    }
    
    AffineMap map = AffineMap::get((*resultOperands).size(), /*symbolCount=*/0, Exprs, b.getContext());   /// stores corresponding AffineMap of above memIVs
    // map.dump();

    ADORA::DataBlockStoreOp newBlockStore = b.create<ADORA::DataBlockStoreOp>\
                (blockstore.getLoc(), blockstore.getSourceMemref(), blockstore.getTargetMemref(), map, *resultOperands);

    newBlockStore.setKernelName(blockstore.getKernelName().str());
    newBlockStore.setId(blockstore.getId().str());

    blockstore.getOperation()->replaceAllUsesWith(newBlockStore);
    blockstore.erase();

    return WalkResult::advance();
  });
}





/////////////////////////////////////////////
//// Base class of emit
/////////////////////////////////////////////
void BaseEmitter::setMapResult(ADORA::KernelOp k, MapperSA* mapper){
  _adg = mapper->getADG();
  KnToConfiguration[k] = Configuration(mapper->_mapping);
}

// @brief Establishes placement constraints for local memory allocations within a given kernel operation.
/// @param kernel A reference to an ADORA::KernelOp object representing the kernel operation.
/// @param mapper A pointer to a MapperSA object used for accessing the data flow graph (DFG) and architecture description graph (ADG).
/// 
/// This function walks through module operations to find local memory allocations that match the specified kernel.
/// For each matching allocation, it identifies the associated I/O nodes and determines which I/O Blocks (IOBs) can access the allocated memory.
/// It then updates the mapper with these placement constraints for proper memory allocation and access during kernel execution.
void BaseEmitter::preestablishPlacementConstraints(ADORA::KernelOp& kernel, MapperSA* mapper){
  DFG* dfg = mapper->getDFG();
  ADG* adg = mapper->getADG();
  int sizeofBank = adg->iobSpadBankSize();
  int dataByte = adg->bitWidth() / 8;

  _moduleop.walk([&](ADORA::LocalMemAllocOp alloc) {  
    if(findElement(alloc.getKernelNameAsStrVector(), kernel.getKernelName()) != -1){
      std::string AllocName = kernel.getKernelName() + ":" + alloc.getId().str();

      if(_LocalAllocToSPMInfo.count(alloc) != 0){
        std::set<int> IONodes = dfg->ioNodes();
        for(auto& id : IONodes){
          DFGIONode* IoNode = dynamic_cast<DFGIONode*>(dfg->node(id));
          if(IoNode->memRefName() == AllocName){
            //// the memory for the ionode has already been allocated
            int bankid = getSPMInfosFromLocalAlloc(alloc).first;
            dfgIoInfo info = getSPMInfosFromLocalAlloc(alloc).second;
            std::vector<int> accessibleIobs = spadBankToIobs(adg, bankid);
            std::vector<ADGNode*> accessibleadgnodes;

            for(auto& elem : adg->nodes()){
              ADGNode* adgNode = elem.second;
              if(adgNode->type() == "IOB"){  
                IOBNode* IobNode = dynamic_cast<IOBNode*>(adgNode);
                if(std::find(accessibleIobs.begin(), accessibleIobs.end(), IobNode->index()) != accessibleIobs.end()){
                  accessibleadgnodes.push_back(adgNode);
                }
              }
            }

            mapper->preestablishPlacementConstraints(dfg->node(id), accessibleadgnodes);
          }
        }  
      }
    }
  });
}


/// @brief Get SPAD information (which bank to transfer data, data size...) for every data block load.
///        This information is store in _LoadToDfgIoInfos/_StoreToDfgIoInfo
/// @param kernel Kernel which has been translated to CDFG
/// @param mapper The SA mapper which has completed mapping
void BaseEmitter::DataBlockOperationsToSPADInfo(ADORA::KernelOp& kernel, MapperSA* mapper){
  // mapper._sched->ioSchedule(mapper._mapping);
  DFG* dfg = mapper->getDFG();
  ADG* adg = mapper->getADG();
  int sizeofBank = adg->iobSpadBankSize();
  int dataByte = adg->bitWidth() / 8;
  
  std::vector<spadBankStatus> bank_status;
  // DenseMap<int, dfgIoInfo> dfg_io_infos;
  bank_status.assign(adg->numIobNodes(), {0, 0, 0, 0}); /*{iob, used, start, end}*/

  std::map<int, dfgIoInfo> dfg_io_infos;
  uint64_t iob_ens = 0;

  /// set occupied banks
  //// TODO: what about _LoadToSPMInfos ??
  for(auto elem : _LocalAllocToSPMInfo){
    ADORA::LocalMemAllocOp alloc = elem.getFirst();
    if(findElement(alloc.getKernelNameAsStrVector(), kernel.getKernelName()) != -1){
      std::string AllocName = kernel.getKernelName() + ":" + alloc.getId().str();
      int selBank = elem.getSecond().first;
      dfgIoInfo ioInfo = elem.getSecond().second;

      std::set<int> IONodes = dfg->ioNodes();
      for(auto& id : IONodes){
        DFGIONode* IoNode = dynamic_cast<DFGIONode*>(dfg->node(id));
        if(IoNode->memRefName() == AllocName){
          int memSize = dynamic_cast<DFGIONode*>(dfg->node(id))->memSize();
          int spadDataByte = adg->cfgSpadDataWidth() / 8; // dual ports of cfg-spad have the same width 
          memSize = (memSize + spadDataByte - 1) / spadDataByte * spadDataByte; // align to spadDataByte
          auto& attr =  mapper->_mapping->dfgNodeAttr(id);
          int iobId = attr.adgNode->id();
          int iobIdx = dynamic_cast<IOBNode*>(adg->node(iobId))->index();
          iob_ens |= 1 << iobIdx; 

          // int selStart = bankStatus[0].second;
          // int selBank = availBanks[0];
          // dfgIoInfo ioInfo = = getSPMInfosFromLocalAlloc(alloc).second;   

          // bank_status[selBank].used = ioInfo.isStore ? 2 : 1; /// Load : 1, Store : 2
          bank_status[selBank].used = 1; /// Load : 1, Store : 2
          bank_status[selBank].iob = iobIdx;
          bank_status[selBank].start = 0;
          bank_status[selBank].end = memSize;  

          dfg_io_infos[id] = ioInfo;
        }
      }
    }
  }


  /// Get io information of every block load ops, including Addr(Spad), iobAddr, LorS
  _moduleop.walk([&](ADORA::DataBlockLoadOp blockload) {  
    if(findElement(blockload.getKernelNameAsStrVector(), kernel.getKernelName()) != -1){
      //// this block load belongs to this kernels 
      std::string BlockLoadName = kernel.getKernelName() + ":" + blockload.getId().str();

      if(_LoadToSPMInfos.count(blockload) != 0){
        //// TODO: what to do?
      }
      else {
        std::set<int> IONodes = dfg->ioNodes();
        for(auto& id : IONodes){
          DFGIONode* IoNode = dynamic_cast<DFGIONode*>(dfg->node(id));
          if(IoNode->memRefName() == BlockLoadName){
            int memSize = dynamic_cast<DFGIONode*>(dfg->node(id))->memSize();
            int spadDataByte = adg->cfgSpadDataWidth() / 8; // dual ports of cfg-spad have the same width 
            memSize = (memSize + spadDataByte - 1) / spadDataByte * spadDataByte; // align to spadDataByte
            auto& attr =  mapper->_mapping->dfgNodeAttr(id);
            int iobId = attr.adgNode->id();
            int iobIdx = dynamic_cast<IOBNode*>(adg->node(iobId))->index();
            iob_ens |= 1 << iobIdx;
            std::vector<int> banks = adg->iobToSpadBanks(iobIdx); // spad banks connected to this IOB
            int minBank = *(std::min_element(banks.begin(), banks.end()));
            std::vector<int> availBanks;
            for(int bank : banks){ // two IOs of the same DFG cannot access the same bank
              if(bank_status[bank].used == 0){
                availBanks.push_back(bank);
              }
            }        

            /// dfg io information 
            int selBank = availBanks[0];
            // int selStart = bankStatus[0].second;

            dfgIoInfo ioInfo;   
            ioInfo.isStore = dfg->getOutNodes().count(id);
            ioInfo.addr = selBank * sizeofBank; /*selBank * sizeofBank + selStart;*/
            ioInfo.iobAddr = ((selBank - minBank) * sizeofBank) / dataByte; /*((selBank - minBank) * sizeofBank + selStart) / dataByte*/;        

            bank_status[selBank].used = ioInfo.isStore ? 2 : 1; /// Load : 1, Store : 2
            bank_status[selBank].iob = iobIdx;
            bank_status[selBank].start = 0;
            bank_status[selBank].end = memSize;  

            dfg_io_infos[id] = ioInfo;

            _LoadToDfgIoInfos[blockload].push_back(ioInfo); 

            _LoadToSPMInfos[blockload].push_back(std::pair(selBank, ioInfo));
          }
        }
      }
    }
  });

  /// Get io information of every block store ops, including Addr(Spad), iobAddr, LorS
  _moduleop.walk([&](ADORA::DataBlockStoreOp blockstore) {  
    if(blockstore.getKernelName() == kernel.getKernelName()){
      std::string BlockStoreName = blockstore.getKernelName().str() + ":" + blockstore.getId().str();

      //// set allocation's info
      Operation* srcOfblockstore = GetTheSourceOperationOfBlockStore(blockstore);

      std::set<int> IONodes = dfg->ioNodes();
      for(auto& id : IONodes){
        DFGIONode* IoNode = dynamic_cast<DFGIONode*>(dfg->node(id));
        if(IoNode->memRefName() == BlockStoreName){
          int memSize = dynamic_cast<DFGIONode*>(dfg->node(id))->memSize();
          int spadDataByte = adg->cfgSpadDataWidth() / 8; // dual ports of cfg-spad have the same width 
          memSize = (memSize + spadDataByte - 1) / spadDataByte * spadDataByte; // align to spadDataByte
          auto& attr =  mapper->_mapping->dfgNodeAttr(id);
          int iobId = attr.adgNode->id();
          int iobIdx = dynamic_cast<IOBNode*>(adg->node(iobId))->index();
          iob_ens |= 1 << iobIdx;
          std::vector<int> banks = adg->iobToSpadBanks(iobIdx); // spad banks connected to this IOB
          int minBank = *(std::min_element(banks.begin(), banks.end()));
          std::vector<int> availBanks;
          for(int bank : banks){ // two IOs of the same DFG cannot access the same bank
            if(bank_status[bank].used == 0){
              availBanks.push_back(bank);
            }
          }        

          /// dfg io information 
          int selBank = availBanks[0];
          // int selStart = bankStatus[0].second;

          dfgIoInfo ioInfo;   
          ioInfo.isStore = dfg->getOutNodes().count(id);
          ioInfo.addr = selBank * sizeofBank; /*selBank * sizeofBank + selStart;*/
          ioInfo.iobAddr = ((selBank - minBank) * sizeofBank) / dataByte; /*((selBank - minBank) * sizeofBank + selStart) / dataByte*/;        

          bank_status[selBank].used = ioInfo.isStore ? 2 : 1; /// Load : 1, Store : 2
          bank_status[selBank].iob = iobIdx;
          bank_status[selBank].start = 0;
          bank_status[selBank].end = memSize;  

          dfg_io_infos[id] = ioInfo;

          assert(_StoreToDfgIoInfo.count(blockstore) == 0 && "One output operation should only be count once.");
          _StoreToDfgIoInfo[blockstore] = ioInfo; 

          if(isa<ADORA::LocalMemAllocOp>(srcOfblockstore) && 
            _LocalAllocToSPMInfo.count(dyn_cast<ADORA::LocalMemAllocOp>(srcOfblockstore)) == 0){
            ADORA::LocalMemAllocOp localAlloc = dyn_cast<ADORA::LocalMemAllocOp>(srcOfblockstore);
            _LocalAllocToSPMInfo[localAlloc] = std::pair(selBank, ioInfo);
          }
        }
      }
    }
  });

  // /// Get io information of every local allocation ops, including Addr(Spad), iobAddr, LorS
  // _moduleop.walk([&](ADORA::LocalMemAllocOp alloc) {  
  //   if(findElement(alloc.getKernelNameAsStrVector(), kernel.getKernelName()) != -1){
  //     std::string AllocName = kernel.getKernelName() + ":" + alloc.getId().str();

  //     if(_LocalAllocToSPMInfo.count(alloc) != 0){
  //       std::set<int> IONodes = dfg->ioNodes();
  //       for(auto& id : IONodes){
  //         DFGIONode* IoNode = dynamic_cast<DFGIONode*>(dfg->node(id));
  //         if(IoNode->memRefName() == AllocName){
  //           int memSize = dynamic_cast<DFGIONode*>(dfg->node(id))->memSize();
  //           int spadDataByte = adg->cfgSpadDataWidth() / 8; // dual ports of cfg-spad have the same width 
  //           memSize = (memSize + spadDataByte - 1) / spadDataByte * spadDataByte; // align to spadDataByte
  //           auto& attr =  mapper->_mapping->dfgNodeAttr(id);
  //           int iobId = attr.adgNode->id();
  //           int iobIdx = dynamic_cast<IOBNode*>(adg->node(iobId))->index();
  //           iob_ens |= 1 << iobIdx; 

  //           /// dfg io information 
  //           int selBank = getSPMInfosFromLocalAlloc(alloc).first;
  //           // int selStart = bankStatus[0].second;

  //           dfgIoInfo ioInfo = = getSPMInfosFromLocalAlloc(alloc).second;   

  //           bank_status[selBank].used = ioInfo.isStore ? 2 : 1; /// Load : 1, Store : 2
  //           bank_status[selBank].iob = iobIdx;
  //           bank_status[selBank].start = 0;
  //           bank_status[selBank].end = memSize;  

  //           dfg_io_infos[id] = ioInfo;

  //           assert(_StoreToDfgIoInfo.count(blockstore) == 0 && "One output operation should only be count once.");
  //           _StoreToDfgIoInfo[blockstore] = ioInfo; 
  //         }
  //       }
  //     }
  //   }
  // });


  _kernel_to_dfg_io_infos[kernel] = dfg_io_infos;
  _kernel_to_iob_ens[kernel] = iob_ens;
}
