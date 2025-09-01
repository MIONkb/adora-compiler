//===----------------------------------------------------------------------===//
//
// Copyright 2023-2024 The ADORA Authors.
//
//===----------------------------------------------------------------------===//
#include "emit/Emit.h"
#include "emit/EmitPytest.h"
#include "emit/OpVisitor.h"
#include "mlir/Dialect/Affine/Utils.h"

#include <ctime>

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;


//===----------------------------------------------------------------------===//
// Some tool functions
//===----------------------------------------------------------------------===//
/// @brief Get a new id for new op which will be emit to C 
/// @param value_name_list
int NewValueNameId(const llvm::SmallDenseMap<mlir::Value, Op_Name_C>& value_name_list){
  int new_id = 0;
  for(auto& elem: value_name_list){
      new_id = new_id < elem.second.id ? elem.second.id : new_id;
  }
  return new_id + 1;
}

void setEmitSkipAttr(mlir::Operation* op){
  op->setAttr("EmitSkip", mlir::UnitAttr::get(op->getContext()));
}

// void emitAffineLoad(AffineLoadOp op) {
//   indent();
//   emitValue(op.getResult());
//   os << " = ";
//   emitValue(op.getMemRef());
//   auto affineMap = op.getAffineMap();
//   AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
//                                   op.getMapOperands());
//   for (auto index : affineMap.getResults()) {
//     os << "[";
//     affineEmitter.emitAffineExpr(index);
//     os << "]";
//   }
//   os << ";";
//   emitInfoAndNewLine(op);
// }

namespace {
class PyOpEmitter : public MLIROpVisitorBase<PyOpEmitter, bool> {
public:
  PyOpEmitter(llvm::raw_ostream &os) : _os(os) {}
  PyOpEmitter(PytestEmitter& emitter, llvm::raw_ostream &os) : 
      _pytestemitter(&emitter) ,_os(os) {setIndent(emitter.getIndent());}
  using MLIROpVisitorBase::visitOp;

  /// Tool functions for emitting
  raw_ostream& indent(){return _os.indent(_indent);}
  void setIndent(unsigned newindent){ _indent = newindent;}

  /// pingpong indicator
  bool _pingpong = false;

  /// @brief emit a new op to python, add this one to op_name_list. 
  /// @param mlirop the corresponding mlir operation
  /// @param type the C type of this operation
  /// @return 
  std::string EmitNewValueAndGetName(mlir::Value v, const std::string type){
    if(_pytestemitter->getValueNameList().count(v)){
      return _pytestemitter->lookupName(v);
    }
    Op_Name_C newopinfo;
    newopinfo.id = NewValueNameId(_pytestemitter->getValueNameList());
    newopinfo.type = type;
    _pytestemitter->appendValueNameList(v, newopinfo);
    return newopinfo.name();
  }

  template <typename opT>
    bool EmitBinary(opT op ,std::string op_symbol){
      std::string type = getEmitType(op.getResult());
      std::string Lhs = _pytestemitter->lookupName(op.getLhs());
      if(Lhs == "")
        Lhs = ConstOpToValueStr[op.getLhs()];
      std::string Rhs = _pytestemitter->lookupName(op.getRhs());
      if(Rhs == "")
        Rhs = ConstOpToValueStr[op.getRhs()];
      assert(Lhs != "" && Rhs != "");

      indent() << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Lhs << " " << op_symbol << " " << Rhs << "\n"; 
      return true;
    }
  ///////////////////////////////
  /// ADORA dialect operations.
  ///////////////////////////////

  /// TODO: 1. support np.array 2.strided blockload 3.mutiple-dimmension list. only support one dimmension list rightnow
  bool visitOp(ADORA::DataBlockLoadOp op) {
    /// DataBlockLoadOp can be seen as a memref subview op, 
    /// for example: 
    /// %0 = ADORA.BlockLoad %arg0 [%arg3, 0, %arg5 * 2, %arg6 * 2] : memref<1x3x230x230xf32> -> memref<1x3x7x62xf32>  {Id = "0", KernelName = "forward_kernel_0"}
    /// 6 variables should be maintained:
    /// DMA_Len: length of one dma request.
    /// DRAM_BaseAddr: the base address in DRAM of the source memref.
    /// DRAM_Offset: the offset address related to affine map
    /// DMA_Request_Offset: keep changing. For this one, DMA_Request_Offset = 230 * i + 230 * 230 * j, i = [0, 7), j = [0, 3)
    /// SPAD_BaseAddr: the base address of Scratchpad memory of the destinated transfer
    /// SPAD_Offset: keep increasing by DMA_Len.
    indent() << "\n";
    indent() << "## " << op << "\n";

    std::string BLid = op.getId().str();
    ::llvm::ArrayRef<int64_t> SourceShape =  op.getOriginalMemrefType().getShape();
    ::llvm::ArrayRef<int64_t> ResultShape =  op.getResultType().getShape();
    // MemRefType SourceType = op.getOriginalMemref();
    // MemRefType ResultType = op.getResultType();
    if(SourceShape.size() == 0){
      /// TODO:
      //// memref<f32> -> memref<2xf32>
      std::string Memref_BaseAddr = _pytestemitter->lookupName(op.getOriginalMemref());
      llvm::SmallVector<dfgIoInfo> DfgIoInfos = _pytestemitter->getDfgIoInfosFromBlockLoad(op);
      assert(DfgIoInfos.size() == 1);
      uint64_t DataBytes = op.getOriginalMemrefType().getElementTypeBitWidth()/8;
      uint64_t DMA_Len = DataBytes;
      uint64_t spadbaddr = DfgIoInfos[0].addr;
      int fuse = 0;
      std::stringstream load_data, spm_ptr;

      load_data << "idata.append(" << Memref_BaseAddr;
      // assert(DRAM_Offset_EachDim.size() == LenEachDim.size());
      // for(int i = 0; i < DRAM_Offset_EachDim.size(); i++){
      //   load_data << "[" << DRAM_Offset_EachDim[i] 
      //             << ":" << DRAM_Offset_EachDim[i] << "+" << LenEachDim[i] << "]";
      // }
      load_data << ")";

      if(_pingpong == true){
        spm_ptr << "iptrs.append(DeviceData(" 
                << "0x" << std::hex << spadbaddr << "+" << std::dec <<DMA_Len
                << " if pingpong else 0x"<< std::hex << spadbaddr
                << ", " << std::dec << DMA_Len 
                << "))";
      }
      else{
        spm_ptr << "iptrs.append(DeviceData(" 
                << "0x" << std::hex << spadbaddr
                << ", " << std::dec<< DMA_Len  
                << "))";
      }     

      indent() << load_data.str() << "\n" ;
      indent() << spm_ptr.str() << "\n\n";

      return true;
    }

    assert(SourceShape.size() == ResultShape.size());

    /// Get DMA_Len 
    uint64_t DataBytes = op.getOriginalMemrefType().getElementTypeBitWidth()/8;
    uint64_t DMA_Len = DataBytes;

    for(int r = ResultShape.size() - 1; r >= 0; r--){
      DMA_Len = DMA_Len * ResultShape[r];
    }
    
    /// Get DRAM_BaseAddr
    std::string Memref_BaseAddr = _pytestemitter->lookupName(op.getOriginalMemref());

    /// Get DRAM_Offset
    std::vector<std::string> DRAM_Offset_EachDim;
    DRAM_Offset_EachDim.resize(ResultShape.size(), "-1");
    /// initialize DRAM_Offset_EachDim
    for(auto elem : DRAM_Offset_EachDim){
      elem = "-1";
    }

    assert(ResultShape.size() == op.getAffineMap().getResults().size());
    for(int exprIdx = 0; exprIdx < op.getAffineMap().getResults().size(); exprIdx++){
      AffineExpr expr = op.getAffineMap().getResult(exprIdx);
      if(expr.getKind() == AffineExprKind::Constant){
        std::string cstValue = std::to_string(expr.dyn_cast<AffineConstantExpr>().getValue());
        DRAM_Offset_EachDim[exprIdx] = cstValue;
      }
      else if(expr.getKind() == AffineExprKind::DimId){
        for(int operandIdx = 0; operandIdx < op.getMapOperands().size(); operandIdx++){
          mlir::Value operand = op.getMapOperands()[operandIdx];
          SmallVector<int>Dimensions = getOperandDimensionsInMap(/*dim=*/operandIdx, /*map=*/op.getAffineMap());
          assert(Dimensions.size() == 1 && "We do not support one index is related to multiple dim of one array.");
          if(Dimensions[0] == exprIdx){
            std::string operandname = _pytestemitter->lookupName(operand);
            DRAM_Offset_EachDim[exprIdx] = operandname;
          }
        }
      }
      else{
        assert(false && "Unsupported Block Access.");
      }
    }


    /// Get DMA_Request_Len from every dim
    std::vector<int64_t> LenEachDim;
    bool continuous = true;
    for(int r = ResultShape.size() - 1; r >= 0; r--){
      LenEachDim.insert(LenEachDim.begin(), ResultShape[r]);
    }    

    /// SPAD_BaseAddr
    /// SPAD_Offset: keep increasing by DMA_Len.
    llvm::SmallVector<dfgIoInfo> DfgIoInfos = _pytestemitter->getDfgIoInfosFromBlockLoad(op);

    llvm::SmallVector<uint64_t> SPAD_BaseAddrs;
    for(dfgIoInfo elem: DfgIoInfos){
      SPAD_BaseAddrs.push_back(elem.addr);
    }

    for(int i = 0; i < SPAD_BaseAddrs.size(); i++)
    {
      auto spadbaddr = SPAD_BaseAddrs[i];
      int fuse = (i == SPAD_BaseAddrs.size() - 1)? 0 : 1; /// fuse: 0 - broadcast, 1 - non-broadcast
      std::stringstream load_data, spm_ptr;
     
      load_data << "idata.append(" << Memref_BaseAddr;
      assert(DRAM_Offset_EachDim.size() == LenEachDim.size());
      for(int i = 0; i < DRAM_Offset_EachDim.size(); i++){
        if(i == 0){
          load_data << "[" ;
        }

        if(op.hasStrides()){
          load_data << DRAM_Offset_EachDim[i] 
                    << ":" << DRAM_Offset_EachDim[i] << "+" << LenEachDim[i] * op.getStridesAsArrayRef()[i]
                    << ":" << op.getStridesAsArrayRef()[i];
        }
        else{
          load_data << DRAM_Offset_EachDim[i] 
                    << ":" << DRAM_Offset_EachDim[i] << "+" << LenEachDim[i];          
        }

        if(i == DRAM_Offset_EachDim.size() - 1){
          load_data << "]";
        }
        else{
          load_data << ",";
        }

      }
      load_data << ")";

      if(_pingpong == true){
        spm_ptr << "iptrs.append(DeviceData(" 
                << "0x" << std::hex << spadbaddr << "+" << std::dec <<DMA_Len
                << " if pingpong else 0x"<< std::hex << spadbaddr
                << ", " << std::dec << DMA_Len 
                << "))";
      }
      else{
        spm_ptr << "iptrs.append(DeviceData(" 
                << "0x" << std::hex << spadbaddr 
                << ", " << std::dec << DMA_Len 
                << "))";
      }

      indent() << load_data.str() << "\n" ;
      indent() << spm_ptr.str() << "\n\n" ;

    }

    return true;
  }

  bool visitOp(ADORA::DataBlockStoreOp op) {
    /// DataBlockStoreOp can be seen as an opposite operation of memref subview op, 
    /// 6 variables should be maintained:
    /// DMA_Len: length of one dma request.
    /// DRAM_BaseAddr: the base address in DRAM of the source memref.
    /// DRAM_Offset: the offset address related to affine map
    /// DMA_Request_Offset: keep changing. For this one, DMA_Request_Offset = 230 * i + 230 * 230 * j, i = [0, 7), j = [0, 3)
    /// SPAD_BaseAddr: the base address of Scratchpad memory of the destinated transfer
    /// SPAD_Offset: keep increasing by DMA_Len.
    indent() << "\n";
    indent() << "## " << op << "\n";

    std::string BLid = op.getId().str();
    ::llvm::ArrayRef<int64_t> SourceShape =  op.getSourceMemrefType().getShape();
    ::llvm::ArrayRef<int64_t> TargetShape =  op.getTargetMemrefType().getShape();

    if(TargetShape.size() == 0){
      //// memref<2xf32> -> memref<f32> 
      std::string Memref_BaseAddr = _pytestemitter->lookupName(op.getTargetMemref());
      // llvm::SmallVector<dfgIoInfo> DfgIoInfos = _pytestemitter->getDfgIoInfosFromBlockLoad(op);
      dfgIoInfo DfgIoInfo = _pytestemitter->getDfgIoInfosFromBlockStore(op);
      // assert(DfgIoInfos.size() == 1);
      uint64_t spadbaddr = DfgIoInfo.addr;
      uint64_t DataBytes = op.getSourceMemrefType().getElementTypeBitWidth()/8;
      uint64_t DMA_Len = DataBytes;
      int fuse = 0;

      std::stringstream store_data, spm_ptr, olen;
      store_data << "odata.append(" << Memref_BaseAddr;
      store_data << ")";


      if(_pingpong == true){
        spm_ptr << "optrs.append(DeviceData(" 
                << "0x" << std::hex << spadbaddr << "+" << std::dec <<DMA_Len
                << " if pingpong else 0x"<< std::hex << spadbaddr
                << ", " << std::dec << DMA_Len 
                << "))";
      }
      else{
        spm_ptr << "optrs.append(DeviceData(" 
                << "0x" << std::hex << spadbaddr
                << ", " << std::dec<< DMA_Len  
                << "))";
      }      
      olen << "olen.append(" << std::dec<< DMA_Len   <<")";

      indent() << store_data.str() << "\n" ;
      indent() << spm_ptr.str() << "\n" ;
      indent() << olen.str() << "\n" ;


      if(IsLastBlockStoreOp(op)){
        if(_pingpong == true){
          // indent() << "stream = runtime.create_stream()\n\n";
          indent() << "await aux_stream_pingpong(\n";
          indent() << "\tstream=stream, config=configs,\n";
          indent() << "\tiptrs=iptrs, idata=idata,\n";
          indent() << "\toptrs=optrs, odata=odata, olen =olen,\n";
          indent() << "\tpingpong=pingpong\n";
          indent() <<")\n\n";
          indent() <<"iptrs.clear(), idata.clear()\n";
          indent() <<"optrs.clear(), odata.clear(), olen.clear()\n\n";
          indent() <<"pingpong = not pingpong\n";
        }
        else{
          indent() << "stream = runtime.create_stream()\n\n";
          indent() << "await aux_stream(\n";
          indent() << "\tstream=stream, config=configs,\n";
          indent() << "\tiptrs=iptrs, idata=idata,\n";
          indent() << "\toptrs=optrs, odata=odata, olen =olen,\n";
          indent() <<")\n\n";
          indent() <<"iptrs.clear(), idata.clear()\n";
          indent() <<"optrs.clear(), odata.clear(), olen.clear()\n\n";     
        }
      }

      return true;
    }

    assert(SourceShape.size() == TargetShape.size());

    /// Get DMA_Len 
    uint64_t DataBytes = op.getSourceMemrefType().getElementTypeBitWidth()/8;
    uint64_t DMA_Len = DataBytes;

    for(int r = SourceShape.size() - 1; r >= 0; r--){
      DMA_Len = DMA_Len * SourceShape[r];
    }
    
    /// Get DRAM_BaseAddr
    std::string Memref_BaseAddr = _pytestemitter->lookupName(op.getTargetMemref());

    /// Get DRAM_Offset
    std::vector<std::string> DRAM_Offset_EachDim;
    DRAM_Offset_EachDim.resize(SourceShape.size(), "-1");
    /// initialize DRAM_Offset_EachDim
    for(auto elem : DRAM_Offset_EachDim){
      elem = "-1";
    }

    assert(TargetShape.size() == op.getAffineMap().getResults().size());
    for(int exprIdx = 0; exprIdx < op.getAffineMap().getResults().size(); exprIdx++){
      AffineExpr expr = op.getAffineMap().getResult(exprIdx);
      if(expr.getKind() == AffineExprKind::Constant){
        std::string cstValue = std::to_string(expr.dyn_cast<AffineConstantExpr>().getValue());
        DRAM_Offset_EachDim[exprIdx] = cstValue;
      }
      else if(expr.getKind() == AffineExprKind::DimId){
        for(int operandIdx = 0; operandIdx < op.getMapOperands().size(); operandIdx++){
          mlir::Value operand = op.getMapOperands()[operandIdx];
          SmallVector<int>Dimensions = getOperandDimensionsInMap(/*dim=*/operandIdx, /*map=*/op.getAffineMap());
          assert(Dimensions.size() == 1 && "We do not support one index is related to multiple dim of one array.");
          if(Dimensions[0] == exprIdx){
            std::string operandname = _pytestemitter->lookupName(operand);
            DRAM_Offset_EachDim[exprIdx] = operandname;
          }
        }
        // SmallVector<int>Dimensions = getOperandDimensionsInMap(/*dim=*/operandIdx, /*map=*/op.getAffineMap());
        // assert(Dimensions.size() == 1);
        // mlir::Value operand = op.getMapOperands()[exprIdx];
      }
      else{
        assert(false && "Unsupported Block Access.");
      }
    }

    std::vector<int64_t> LenEachDim;
    bool continuous = true;
    for(int r = SourceShape.size() - 1; r >= 0; r--){
      LenEachDim.insert(LenEachDim.begin(), SourceShape[r]);
    }    


    /// SPAD_BaseAddr
    /// SPAD_Offset: keep increasing by DMA_Len.
    dfgIoInfo DfgIoInfos = _pytestemitter->getDfgIoInfosFromBlockStore(op);
    llvm::SmallVector<uint64_t> SPAD_BaseAddrs;
    SPAD_BaseAddrs.push_back(DfgIoInfos.addr);

    /// Emit pytest
    for(int i = 0; i < SPAD_BaseAddrs.size(); i++)
    {
      auto spadbaddr = SPAD_BaseAddrs[i];
      int fuse = (i == SPAD_BaseAddrs.size() - 1)? 0 : 1; /// fuse: 0 - broadcast, 1 - non-broadcast
      std::stringstream store_data, spm_ptr, olen;

      store_data << "odata.append(" << Memref_BaseAddr;
      assert(DRAM_Offset_EachDim.size() == LenEachDim.size());
      for(int i = 0; i < DRAM_Offset_EachDim.size(); i++){
        if(i == 0){
          store_data << "[" ;
        }

        if(op.hasStrides()){
          store_data << DRAM_Offset_EachDim[i] 
                  << ":" << DRAM_Offset_EachDim[i] << "+" << LenEachDim[i] * op.getStridesAsArrayRef()[i] 
                  << ":" << op.getStridesAsArrayRef()[i];
        }
        else{
          store_data << DRAM_Offset_EachDim[i] 
                  << ":" << DRAM_Offset_EachDim[i] << "+" << LenEachDim[i];
        }
        if(i == DRAM_Offset_EachDim.size() - 1){
          store_data << "]";
        }
        else{
          store_data << ",";
        }
      }
      store_data << ")";
      if(_pingpong == true){
        spm_ptr << "optrs.append(DeviceData(" 
                << "0x" << std::hex << spadbaddr << "+" << std::dec <<DMA_Len
                << " if pingpong else 0x"<< std::hex << spadbaddr
                << ", " << std::dec << DMA_Len 
                << "))";
      }
      else{
        spm_ptr << "optrs.append(DeviceData(" 
                << "0x" << std::hex << spadbaddr 
                << ", " << std::dec << DMA_Len 
                << "))";
      }
      
      olen << "olen.append(" << DMA_Len <<")";

      indent() << store_data.str() << "\n" ;
      indent() << spm_ptr.str() << "\n";
      indent() << olen.str() << "\n\n";

    }

    // indent() << "\n";

    if(IsLastBlockStoreOp(op)){
      if(_pingpong == true){
        // indent() << "stream = runtime.create_stream()\n\n";
        indent() << "await aux_stream_pingpong(\n";
        indent() << "\tstream=stream, config=configs,\n";
        indent() << "\tiptrs=iptrs, idata=idata,\n";
        indent() << "\toptrs=optrs, odata=odata, olen =olen,\n";
        indent() << "\tpingpong=pingpong\n";
        indent() <<")\n\n";
        indent() <<"iptrs.clear(), idata.clear()\n";
        indent() <<"optrs.clear(), odata.clear(), olen.clear()\n\n";
        indent() <<"pingpong = not pingpong\n";
      }
      else{
        indent() << "stream = runtime.create_stream()\n\n";
        indent() << "await aux_stream(\n";
        indent() << "\tstream=stream, config=configs,\n";
        indent() << "\tiptrs=iptrs, idata=idata,\n";
        indent() << "\toptrs=optrs, odata=odata, olen =olen,\n";
        indent() <<")\n\n";
        indent() <<"iptrs.clear(), idata.clear()\n";
        indent() <<"optrs.clear(), odata.clear(), olen.clear()\n\n";     
      }
    }

    return true;    
  }

  bool visitOp(ADORA::LocalMemAllocOp op) {
    /// %2 = ADORA.LocalMemAlloc memref<20x20xi32>  {Id = "2", KernelName = "IntVecAdd"}
    indent() << "\n";
    indent() << "## " << op << "\n";

    std::string BLid = op.getId().str();
    ::llvm::ArrayRef<int64_t> ResultShape =  op.getResultType().getShape();

    uint64_t DataBytes = op.getResultType().getElementTypeBitWidth()/8;
    uint64_t Len = DataBytes;
    for(int r = ResultShape.size() - 1; r >= 0; r--){
        Len = Len * ResultShape[r];
    }

    dfgIoInfo DfgIoInfos = _pytestemitter->getSPMInfosFromLocalAlloc(op).second;
    uint64_t SPAD_BaseAddr = DfgIoInfos.addr;

    std::stringstream alloc_ptr;

    if(_pingpong == true){
      alloc_ptr << "data_ptr.append(DeviceData(" 
                << "0x" << std::hex << SPAD_BaseAddr << "+" << std::dec <<Len
                << " if pingpong else 0x"<< std::hex << SPAD_BaseAddr
                << ", " << std::dec << Len
                << "))";
    }
    else{
      alloc_ptr << "data_ptr.append(DeviceData(" 
              << "0x" << std::hex << SPAD_BaseAddr
              << ", " << std::dec << Len
              << "))";
    }
    
    indent() << alloc_ptr.str() << "\n";

    return true;
  }
  


  bool visitOp(ADORA::KernelOp op) {
    if(op.getOperation()->hasAttr("EmitSkip")){
      return true;
    }
    indent() << "\n";
    if(!MapHasKey(_pytestemitter->KnToCfgExe, op)){
      // Configuration cfg = ;
      ADG* adg = _pytestemitter->getADG();
      _pytestemitter->GenerateCGRACFGAndEXE(op, _pytestemitter->KnToConfiguration[op], adg);
    }
    if(!op.getKernelName().empty()){
      indent() << "### " << op.getKernelName() << "\n";
    }

    std::vector<std::string> strs = split_str_by_char(_pytestemitter->KnToCfgExe[op], '\n');
    for(std::string str: strs){
      indent() << str << "\n";
    }
    
    indent() << "\n\n";
    return true;
  }

  ///////////////////////////////
  /// ADORA Tensor dialect operations.
  ///////////////////////////////
  bool visitOp(ADORA::ADORATensor::GemmOp gemmop) {
    mlir::Operation* op = gemmop.getOperation()->getNextNode();
    // if(isa<affine::AffineForOp>(op)){
    //   op->setAttr("ADORAGemm", mlir::UnitAttr::get(op->getContext()));
    // }
    indent() << "### GemmOp: " << gemmop << "\n";
    indent() << "pingpong = True" << "\n";
    indent() << "stream = runtime.create_stream()" << "\n";

    _pingpong = true;
    if(isa<mlir::affine::AffineForOp>(op) && op->hasAttr("ADORAGemm")){
      visitOp(dyn_cast<mlir::affine::AffineForOp>(op));
    }

    _pingpong = false;
    setEmitSkipAttr(op);

    indent() << "### End of GemmOp: " << gemmop << "\n";
  }

  // bool visitOp(BufferOp op) {
  //   if (op.getDepth() == 1)
  //     return emitter.emitAlloc(op), true;
  //   return op.emitOpError("only support depth of 1"), false;
  // }
  // bool visitOp(ConstBufferOp op) { return emitter.emitConstBuffer(op), true; }
  // bool visitOp(StreamOp op) { return emitter.emitStreamChannel(op), true; }
  // bool visitOp(StreamReadOp op) { return emitter.emitStreamRead(op), true; }
  // bool visitOp(StreamWriteOp op) { return emitter.emitStreamWrite(op), true; }
  // bool visitOp(AxiBundleOp op) { return true; }
  // bool visitOp(AxiPortOp op) { return emitter.emitAxiPort(op), true; }
  // bool visitOp(AxiPackOp op) { return false; }
  // bool visitOp(PrimMulOp op) { return emitter.emitPrimMul(op), true; }
  // bool visitOp(PrimCastOp op) { return emitter.emitAssign(op), true; }
  // bool visitOp(hls::AffineSelectOp op) {
  //   return emitter.emitAffineSelect(op), true;
  // }

  /// Function operations.
  // bool visitOp(func::CallOp op) { return emitter.emitCall(op), true; }
  bool visitOp(memref::AllocaOp op) { 
    mlir::MemRefType mt = op.getType();
    assert(mt.getShape().size() == 0);
    mlir::Type t = mt.getElementType();

    std::string type = getEmitType(t);
    indent() << EmitNewValueAndGetName(op.getResult(), type) << "\n";

    return true; 
  }
  bool visitOp(func::ReturnOp op) { return true; }

  /// SCF statements.
  // bool visitOp(scf::ForOp op) { return emitter.emitScfFor(op), true; };
  // bool visitOp(scf::IfOp op) { return emitter.emitScfIf(op), true; };
  // bool visitOp(scf::ParallelOp op) { return false; };
  // bool visitOp(scf::ReduceOp op) { return false; };
  // bool visitOp(scf::ReduceReturnOp op) { return false; };
  // bool visitOp(scf::YieldOp op) { return emitter.emitScfYield(op), true; };

  /// CF 
  bool visitOp(cf::BranchOp op) { 
    _pytestemitter->emitBlock(*(op.getDest()), _os);
    return true;
  }

  /// Affine statements.
  bool visitOp(affine::AffineForOp op) { 
    if(op.getOperation()->hasAttr("EmitSkip")){
      return true;
    }

    indent() << "for ";
    auto iterVar = op.getInductionVar();
    assert(op.getLowerBoundMap().getResults().size()==1);
    _os << EmitNewValueAndGetName(iterVar, "int") << " in range(";
    _os << op.getLowerBoundMap().getResult(0) << ", ";

    // Emit loop invariant(upper bound)
    assert(op.getUpperBoundMap().getResults().size()==1);
    _os << op.getUpperBoundMap().getResult(0) << ", ";

    // Emit loop step
    _os  << op.getStep() << "):\n";

    if(op.getOperation()->hasAttr("ADORAGemm")){
      _pytestemitter->emitGemmBlock(*(op.getBody()), _os);
    }
    else{
      _pytestemitter->emitBlock(*(op.getBody()), _os);
    }

    _os << "\n";
    return true;
  }

  // bool visitOp(AffineIfOp op) { return emitter.emitAffineIf(op), true; }
  // bool visitOp(AffineParallelOp op) {
  //   return emitter.emitAffineParallel(op), true;
  // }
  // bool visitOp(AffineApplyOp op) { return emitter.emitAffineApply(op), true; }
  // bool visitOp(AffineMaxOp op) {
  //   return emitter.emitAffineMaxMin(op, "max"), true;
  // }
  // bool visitOp(AffineMinOp op) {
  //   return emitter.emitAffineMaxMin(op, "min"), true;
  // }
  // bool visitOp(AffineLoadOp op) { 
    // return 
    // return emitter.emitAffineLoad(op), true; 
  // }
  bool visitOp(::mlir::affine::AffineStoreOp op) { 
    // std::string type = getEmitType(op.getResult());
    std::string value;
    if(isa<LLVM::UndefOp>(op.getValue().getDefiningOp())){
      value = "0";
    }
    else{
      std::string value = _pytestemitter->lookupName(op.getValue());
      // assert("Unsupported!\n");
    }

    assert(op.getMemref().getType().cast<MemRefType>().getShape().size() == 0);
    std::string memref = _pytestemitter->lookupName(op.getMemref());
    indent() << memref << " = " << value << "\n";
    return true;
    // return emitter.emitAffineStore(op), true; 
  }
  // bool visitOp(AffineVectorLoadOp op) { return false; }
  // bool visitOp(AffineVectorStoreOp op) { return false; }
  bool visitOp(affine::AffineYieldOp op) { return true; }

  /// Vector statements.
  // bool visitOp(vector::TransferReadOp op) {
  //   return emitter.emitTransferRead(op), true;
  // };
  // bool visitOp(vector::TransferWriteOp op) {
  //   return emitter.emitTransferWrite(op), true;
  // };
  // bool visitOp(vector::BroadcastOp op) {
  //   return emitter.emitBroadcast(op), true;
  // };

  // /// Memref statements.
  // bool visitOp(memref::AllocOp op) { return emitter.emitAlloc(op), true; }
  // bool visitOp(memref::AllocaOp op) { return emitter.emitAlloc(op), true; }
  // bool visitOp(memref::LoadOp op) { return emitter.emitLoad(op), true; }
  // bool visitOp(memref::StoreOp op) { return emitter.emitStore(op), true; }
  // bool visitOp(memref::DeallocOp op) { return true; }
  // bool visitOp(memref::CopyOp op) { return emitter.emitMemCpy(op), true; }
  // bool visitOp(memref::ReshapeOp op) { return emitter.emitReshape(op), true;
  // } bool visitOp(memref::CollapseShapeOp op) {
  //   return emitter.emitReshape(op), true;
  // }
  // bool visitOp(memref::ExpandShapeOp op) {
  //   return emitter.emitReshape(op), true;
  // }
  // bool visitOp(memref::ReinterpretCastOp op) {
  //   return emitter.emitReshape(op), true;
  // }

  /// Arithmetic dialect
  bool visitOp(arith::ConstantOp op) {
    // This indicates the constant type is scalar (float, integer, or bool).
    // if (isDeclared(op.getResult()))
    //   return;
    // arith::ConstantOp constin = dyn_cast<arith::ConstantOp>(in);
    // indent();
    mlir::Attribute constattr = op.getOperation()->getAttr(op.getValueAttrName());

    if(isa<FloatAttr>(constattr)){
      FloatAttr floatattr = dyn_cast<FloatAttr>(constattr);
      if(floatattr.getType().isF64()){
        double value = floatattr.getValueAsDouble();
        ConstOpToValueStr[op.getResult()] = std::to_string(value);  
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "double");
        // _os << "double " << name_c << " = " << std::to_string(value) << ";\n";
      } 
      else if(floatattr.getType().isF32()){
        double value = floatattr.getValueAsDouble();
        ConstOpToValueStr[op.getResult()] = std::to_string(value);  
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "float");
        // _os << "float " << name_c << " = " << std::to_string(value) << ";\n";
      }
    } 
    else if(isa<IntegerAttr>(constattr))
    {
      IntegerAttr intattr = dyn_cast<IntegerAttr>(constattr);
      if(intattr.getType().isInteger(16)){    
        int value = intattr.getInt();  
        ConstOpToValueStr[op.getResult()] = std::to_string(value);   
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "int16_t");
        // _os << "int16_t " << name_c << " = " << std::to_string(value) << ";\n";
      }
      else if(intattr.getType().isInteger(32)){
        int value = intattr.getInt();   
        ConstOpToValueStr[op.getResult()] = std::to_string(value);  
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "int32_t");        
        // _os << "int32_t " << name_c << " = " << std::to_string(value) << ";\n";
      }
      else if(intattr.getType().isInteger(64)){ 
        int value = intattr.getInt();   
        ConstOpToValueStr[op.getResult()] = std::to_string(value);  
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "int64_t");
        // _os << "int64_t " << name_c << " = " << std::to_string(value) << ";\n";
      }
      else if(intattr.getType().isUnsignedInteger(16)){    
        int value = intattr.getInt();   
        ConstOpToValueStr[op.getResult()] = std::to_string(value);  
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "uint16_t");
        // _os << "uint16_t " << name_c << " = " << std::to_string(value) << ";\n";
      }
      else if(intattr.getType().isUnsignedInteger(32)){
        int value = intattr.getInt();   
        ConstOpToValueStr[op.getResult()] = std::to_string(value);  
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "uint32_t");
        // _os << "uint32_t " << name_c << " = " << std::to_string(value) << ";\n";
      }
      else if(intattr.getType().isUnsignedInteger(64)){ 
        int value = intattr.getInt(); 
        ConstOpToValueStr[op.getResult()] = std::to_string(value);  
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "uint64_t");
        // _os << "uint64_t " << name_c << " = " << std::to_string(value) << ";\n";
      }
      else if(intattr.getType().isIndex()){ 
        int value = intattr.getInt();   
        ConstOpToValueStr[op.getResult()] = std::to_string(value);
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "int");
        // _os << "int " << name_c << " = " << std::to_string(value) << ";\n";
      }
    }
    else if(isa<BoolAttr>(constattr))
    {
      BoolAttr boolattr = dyn_cast<BoolAttr>(constattr);
      bool value = boolattr.getValue(); 
      ConstOpToValueStr[op.getResult()] = std::to_string(value);
      // std::string name_c = EmitNewValueAndGetName(op.getResult(), "bool");
      // _os << "bool " << name_c << " = " << std::to_string(value) << ";\n";
    }
    else if (auto denseAttr = op.getValue().dyn_cast<DenseElementsAttr>()) {
      // indent();
      denseAttr.dump();
      op.emitError("has unsupported constant denseAttr type.");
      abort();
      // emitArrayDecl(op.getResult());
      // os << " = {";
      // auto type =
      //   op.getResult().getType().template cast<ShapedType>().getElementType();

      // unsigned elementIdx = 0;
      // for (auto element : denseAttr.template getValues<Attribute>()) {
      //   auto string = getConstantString(type, element);
      //   if (string.empty())
      //     op.emitOpError("constant has invalid value");
      //   os << string;
      //   if (elementIdx++ != denseAttr.getNumElements() - 1)
      //     os << ", ";
      // }
      // os << "};";
      // emitInfoAndNewLine(op);
    } else
      op.emitError("has unsupported constant type.");
    
    // ConstOpToValueStr_print();
  }

  bool visitOp(arith::AddIOp op) {
      std::string type = getEmitType(op.getResult());
      std::string Lhs = _pytestemitter->lookupName(op.getLhs());
      if(Lhs == "")
        Lhs = ConstOpToValueStr[op.getLhs()];
      std::string Rhs = _pytestemitter->lookupName(op.getRhs());
      if(Rhs == "")
        Rhs = ConstOpToValueStr[op.getRhs()];
      assert(Lhs != "" && Rhs != "");

      if( Lhs == "0" && Rhs == "0")
        indent() << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << "0" << "\n";
      else if( Lhs == "0" && Rhs != "0")
        indent() << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Rhs << "\n";
      else if( Lhs != "0" && Rhs == "0")
        indent() << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Lhs << "\n";
      else
        indent()  << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Lhs << " " << "+" << " " << Rhs << "\n"; 
      
      return true;
    // return EmitBinary(op, "+");
  }

  bool visitOp(arith::SubIOp op) {
      std::string type = getEmitType(op.getResult());
      std::string Lhs = _pytestemitter->lookupName(op.getLhs());
      if(Lhs == "")
        Lhs = ConstOpToValueStr[op.getLhs()];
      std::string Rhs = _pytestemitter->lookupName(op.getRhs());
      if(Rhs == "")
        Rhs = ConstOpToValueStr[op.getRhs()];
      assert(Lhs != "" && Rhs != "");

      if( Lhs == "0" && Rhs == "0")
        indent() << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << "0" << "\n";
      else if( Lhs != "0" && Rhs == "0")
        indent() << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Lhs << "\n";
      else
        indent() << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Lhs << " " << "-" << " " << Rhs << ";\n"; 
      
      return true;

    // return EmitBinary(op, "-");
  }

  bool visitOp(arith::MulIOp op) {
    return EmitBinary(op, "*");
  }

  bool visitOp(arith::DivSIOp op) {
    return EmitBinary(op, "/");
  }

  bool visitOp(LLVM::UndefOp op) {
    // op.getRes()
    // return EmitBinary(op, "/");
    return true; /// skip 
  }

  void ConstOpToValueStr_print() {
    llvm::errs()<< "=======Print ConstOpToValueStr=======\n";
    for(auto elem : ConstOpToValueStr){
      llvm::errs() << elem.first << "  --->  " << elem.second << "\n";
    }
    llvm::errs()<< "=======End Print=======\n";
  }

private:
  PytestEmitter* _pytestemitter;
  llvm::DenseMap<mlir::Value, std::string> ConstOpToValueStr;
  llvm::raw_ostream &_os;
  unsigned _indent = 0;
};
PyOpEmitter* opEmitter;
} // namespace


// static InFlightDiagnostic emitError(Operation *op, const Twine &message) {
//   // state.encounteredError = true;
//   return op->emitError(message);
// }
//===----------------------------------------------------------------------===//
// Members of PythonEmitter class
//===----------------------------------------------------------------------===//
/// @brief Emit the header of a function including function name, function args...
/// @param os
void PytestEmitter::emitFunctionHead(func::FuncOp &funcop, llvm::raw_ostream &os) {
  std::stringstream ostr;
  ostr << "async def " << funcop.getSymName().str() << "("
       << "runtime: DeviceRuntime, ";

  // Funtion args
  ArrayRef<mlir::Type> argTypes = funcop.getArgumentTypes();
  for(int argIdx = 0; argIdx < argTypes.size(); argIdx++){
    if(argIdx != 0){
      ostr << ", ";
    }
    mlir::Type argType = argTypes[argIdx];

    // argType.dump();
    Op_Name_C arg_info("arg", argIdx);
    appendValueNameList(funcop.getBody().getArgument(argIdx), arg_info);
    if(argType.isa<MemRefType>()){
      ostr <<"arg_" << argIdx  << ": ndarray" ;
    } 
    else if(argType.isIndex()){
      ostr << "arg_" << argIdx  << ": int";
    }
    else if(argType.isInteger(32)){
      ostr  << "arg_" << argIdx << ": int";
    }
    else if(argType.isInteger(64)){
      ostr  << "arg_" << argIdx << ": int";
    }
    else if(argType.isUnsignedInteger(32)){
      ostr  << "arg_" << argIdx << ": int";
    }
    else if(argType.isUnsignedInteger(64)){
      ostr  << "arg_" << argIdx << ": int";
    }
    
  }
  ostr << "):\n";

  ostr << "    # runtime.log.info(\"[ADORA] Starting CGRA call (" 
       << funcop.getSymName().str() << ")\")\n";

  ostr << "    iptrs, idata = [],[]\n" 
       << "    optrs, odata, olen = [],[],[]\n" 
       << "    configs, data_ptr = [],[]\n";

  os << ostr.str();
}

/// @brief Emit a block (maybe a loop body, maybe a function body), especially the for loop structure
/// @param os
void PytestEmitter::emitBlock(mlir::Block &block, llvm::raw_ostream &os) {
  // std::stringstream ostr;
  addIndent();

  opEmitter->setIndent(getIndent());
  block.dump();

  for (auto &op : block) {
    op.dump();
    // TypeSwitch<Operation *, bool>(&op)
    //   .template Case<
    //     // Affine statements.
    //     affine::AffineForOp,
    //     // // Special expressions.
    //     arith::ConstantOp
    //   >([&](auto opNode) -> bool {
    //     op.dump();
    //     return true;
    //   })
    //   .Default([&](auto opNode) -> bool {
    //     llvm::errs() << "No support!\n";
    //     return false;
    //   });
    if(opEmitter->dispatchVisitor(&op)){
      continue;
    }
    else{
      op.emitError("can't be correctly emitted.");
    } 
  }
  reduceIndent();
  opEmitter->setIndent(getIndent());
  // os << ostr.str();
}


/// @brief Emit the whole module op to CGRA Call function in C languange
/// @param os
/// @return Successful or not
bool PytestEmitter::emitPytest(llvm::raw_ostream &os) {
  opEmitter = new PyOpEmitter(*this, os);
  // ADORAEmitterState state(os);
  // ModuleEmitter(state).emitModule(module);
  // return failure(state.encounteredError);
  /// get time
  std::time_t t = std::time(nullptr);
  std::tm tm;
  #ifdef _WIN32
  localtime_s(&tm, &t);
  #else
  localtime_r(&t, &tm);
  #endif
  char timebuf[64];
  std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", &tm);

  os << R"XXX(
"""
Copyright (c) 2025 ADORA
All rights reserved.
Automatically generated file for pytest/cocotb based CGRA call function from ADORA.
)XXX";  
  os << "Generated on: " << timebuf << "\n";
  os << R"XXX(
"""
from test_runif import DeviceData, DeviceConfig, DeviceStream, DeviceRuntime
from typing import List
from numpy import ndarray

async def aux_stream(
    stream: DeviceStream, config: List[DeviceConfig], 
    iptrs: List[DeviceData], idata: List, 
    optrs: List[DeviceData], odata: List, olen: List):
    """
    Execute a device stream workflow.

    Parameters
    ----------
    stream : DeviceStream
        The device stream instance to operate on.
    config : List[DeviceConfig]
        Configuration objects to apply before execution.
    iptrs : List[DeviceData]
        Device pointers for input buffers.
    idata : List
        Host-side input data corresponding to `iptrs`.
    optrs : List[DeviceData]
        Device pointers for output buffers.
    odata : List
        Host-side output data containers corresponding to `optrs`.
    olen : List[int]
        Expected output lengths for each output buffer.
    """
    # ------------------------------
    # 1. Apply stream configuration
    # ------------------------------
    await stream.apply(config)
    await stream.config(config_id=0)
    # ------------------------------
    # 2. Host -> Device transfer
    # ------------------------------
    for i in range(len(iptrs)):
        await stream.memcpyHostToDevice(d_data=iptrs[i], h_data=idata[i], size=len(idata[i]))
    # ------------------------------
    # 3. Execute on device
    # ------------------------------
    await stream.execution_start()
    await stream.execution_finish()
    # ------------------------------
    # 4. Device → Host transfer
    # ------------------------------
    for i in range(len(optrs)):
        await stream.memcpyDeviceToHost(d_data=optrs[i], h_data=odata[i], size=olen[i], dtype='i')

    ## await stream.release()
    return

  
async def aux_stream_pingpong(
    stream: DeviceStream, config: List[DeviceConfig], 
    iptrs: List[DeviceData], idata: List[ndarray], 
    optrs: List[DeviceData], odata: List, olen: List, 
    pingpong: bool):
    """
    Execute a device stream workflow.

    Parameters
    ----------
    stream : DeviceStream
        The device stream instance to operate on.
    config : List[DeviceConfig]
        Configuration objects to apply before execution.
    iptrs : List[DeviceData]
        Device pointers for input buffers.
    idata : List
        Host-side input data corresponding to `iptrs`.
    optrs : List[DeviceData]
        Device pointers for output buffers.
    odata : List
        Host-side output data containers corresponding to `optrs`.
    olen : List[int]
        Expected output lengths for each output buffer.
    pingpong : bool
        Indicates the pingpong phase(ping-phase or pong-phase)
    """
    # ------------------------------
    # 1. Apply stream configuration
    # ------------------------------     
    await stream.config(config_id=0)
    # ------------------------------
    # 2. Host -> Device transfer
    # ------------------------------
    for i in range(len(iptrs)):
        if(pingpong == 0):
            await stream.memcpyHostToDevice(d_data=iptrs[i], h_data=idata[i], size=len(idata[i]))
        else:
            await stream.memcpyHostToDevice(d_data=iptrs[i]+len(idata[i]), h_data=idata[i], size=len(idata[i]))

    # ------------------------------
    # 3. Execute on device
    # ------------------------------
    await stream.execution_start()
    await stream.execution_finish()
    # ------------------------------
    # 4. Device → Host transfer
    # ------------------------------
    for i in range(len(optrs)):
        if(pingpong == 0):
            await stream.memcpyDeviceToHost(d_data=optrs[i], h_data=odata[i], size=olen[i])
        else :
            await stream.memcpyDeviceToHost(d_data=optrs[i]+olen[i], h_data=odata[i], size=olen[i])
    
    # await stream.synchronize()

    await stream.release()
    return

async def aux_stream_pingpong_init(
    stream: DeviceStream, config: List[DeviceConfig]
    ):
    """
    Apply stream configuration
    """
    await stream.apply(config)  

    await stream.release()
    return

## ===----------------------------------------------------------------------===//
## Configuration Data 
## ===----------------------------------------------------------------------===//
)XXX";  

  for(auto elem : KnToCfgData){
    os << elem.second << "\n";
  }

  // _moduleop.walk([&](mlir::Operation* op) {
  //   op->dump();
  // });
  /// Emit module
  for(auto funcop : _moduleop.getOps<func::FuncOp>()){
    funcop.dump();
    /// function head
    emitFunctionHead(funcop, os);

    /// function body
    // addIndent();
    emitBlock(funcop.getBody().front(), os);

    // / function tail
    os << R"XXX(
    await stream.synchronize()
)XXX";
  }

  delete opEmitter;
}

bool PytestEmitter::emitCGRACallFunction(llvm::raw_ostream &os) {
  emitPytest(os);
}

// Traverse the whole module to find which operation keeps a "VAR_CONFIG" attr
// equal to arg config, and return the result value of the operation
std::string PytestEmitter::lookupVarConfigName(const std::string config){ 
  mlir::Value target_value;
  _moduleop.walk([&](mlir::Operation* op)-> WalkResult {
    op->dump();
    if(op->hasAttr("VAR_CONFIG") 
        && config == dyn_cast<StringAttr>(op->getAttr("VAR_CONFIG")).str()){
      /// find the value crresponding to the config
      assert(op->getResults().size() == 1);
      target_value = op->getResult(0);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  std::string target_name = lookupName(target_value);
  assert(target_name != "");
  return target_name;
}

/// @brief Get the config data
std::string PytestEmitter::GenerateCGRAConfig(
  ADORA::KernelOp& kernel, Configuration cfg, ADG* adg){
  // adg->print();
  std::string CFGarrayName = "cfgbit_" + kernel.getKernelName();
  std::stringstream CFGdata;

   // cfg.dumpCfgData(std::cout);
  std::map<int, dfgIoInfo> dfg_io_infos = std::move(_kernel_to_dfg_io_infos[kernel]);
  for(auto &elem : dfg_io_infos){
    cfg.setDfgIoSpadAddr(elem.first, elem.second.iobAddr);
  }
  std::vector<CfgDataPacket> cfgData;
  cfg.getCfgData(cfgData);
  // cfg.getCfgData(cfgData); /// debug
  int cfgSpadDataByte = adg->cfgSpadDataWidth() / 8;
  int cfgAddrWidth = adg->cfgAddrWidth();
  int cfgDataWidth = adg->cfgDataWidth();
  int alignWidth = (cfgAddrWidth > 16) ? 32 : 16;
  assert(alignWidth >= cfgAddrWidth && cfgDataWidth >= alignWidth);
  int cfgNum = 0;
  for(auto& cdp : cfgData){
    cfgNum += cdp.data.size() * 32 / cfgDataWidth;
  }

  // if(cfgAddrWidth > 16){
  //   CFGdata << "volatile unsigned int ";
  // }else{
  //   CFGdata << "volatile unsigned short ";
  // }
  CFGdata << "\"\"\" kernel: " << kernel.getKernelName()
          << ",  cfgNum: " << cfgNum << "\"\"\"\n";
  CFGdata << CFGarrayName << " = [\n";
  CFGdata << std::hex;
  int alignWidthHex = alignWidth/4;
  for(auto& cdp : cfgData){
    CFGdata << "\t\t";
    for(auto data : cdp.data){      
      if(alignWidth == 32){
         CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << data << ", ";
      }else{
        CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (data & 0xffff) << ", ";
        CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (data >> 16) << ", ";
      }
            
    }
    CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (cdp.addr) << ",\n";
  }
  CFGdata << std::dec << "\t]\n\n";

  KnToCfgData[kernel] = CFGdata.str();
  KnToCfgArrayInfo[kernel] = std::pair(CFGarrayName, cfgNum);

  return CFGdata.str();
}

/// @brief Get the config data
std::string PytestEmitter::GenerateCGRAConfig(
  ADORA::KernelOp& kernel, MapperSA* mapper){
  ADG* adg = mapper->getADG();
  Configuration cfg(mapper->_mapping);
  KnToConfiguration[kernel] = cfg;
  _adg = adg;
  return GenerateCGRAConfig(kernel, cfg, adg);
}



/// @brief Get the config and execution instructions of CGRA
void PytestEmitter::GenerateCGRACFGAndEXE(
  ADORA::KernelOp& kernel, Configuration cfg, ADG* adg){
  // adg->print();
  /////////////////////// CFG is generated in GenerateCGRACFGData()
  std::stringstream CFGandEXE;

  int cfgSpadDataByte = adg->cfgSpadDataWidth() / 8;
  int cfgAddrWidth = adg->cfgAddrWidth();
  int cfgDataWidth = adg->cfgDataWidth();
  int alignWidth = (cfgAddrWidth > 16) ? 32 : 16;
  assert(alignWidth >= cfgAddrWidth && cfgDataWidth >= alignWidth);

  if(!MapHasKey(KnToCfgData, kernel)){
    std::string cfgdata = GenerateCGRAConfig(kernel, cfg, adg);
    CFGandEXE << cfgdata << "\n";
  }
  assert(MapHasKey(KnToCfgData, kernel));
  std::string CFGarrayName = KnToCfgArrayInfo[kernel].first;
  int cfgNum = KnToCfgArrayInfo[kernel].second;

  /// replace the variable part of the config
  //// TODO: what about variable runtime
  CFGandEXE << std::hex;
  for(auto elem : cfg.VarReplaceInfo){
    std::string varcfg_name = lookupVarConfigName(elem.first);
    for(auto& replace: elem.second) {
      CFGandEXE << std::dec << CFGarrayName << "[" << replace.Idx0 << "][" << replace.Idx1 << "] = ";
      assert(replace.lshift == 0 || replace.rshift == 0);
      if(replace.lshift == 0){
        // right shift
        CFGandEXE << std::hex << "(" << varcfg_name << " >> 0x" << replace.rshift << ")"
                  << " | " 
                  << std::dec << "(" << CFGarrayName  <<"[" << replace.Idx0 << "][" << replace.Idx1 << "]" 
                  << std::hex << " & 0x" << replace.getMask() << ")\n" ;
      }
      else{
        // left shift
        CFGandEXE << std::hex << "(" << varcfg_name << " << 0x" << replace.lshift << ")"
                  << " | " 
                  << std::dec << "(" << CFGarrayName <<"[" << replace.Idx0 << "][" << replace.Idx1 << "]" 
                  << std::hex << " & 0x" << replace.getMask() << ")\n" ;
      }
    }
  }
  CFGandEXE << std::dec;
 
    // _cfg_num = cfgNum;
    // _cfg_len = cfgNum * (alignWidth + cfgDataWidth) / 8; // length of config_addr and config_data in bytes
    // int cfgSpadSize = _adg->cfgSpadSize();
    // int cfgBaseAddr;
    // _ld_cfg_dep = 0;
    // if(_cfg_len <= cfgSpadSize - _old_cfg_status.end){
    //     cfgBaseAddr = _old_cfg_status.end;
    // }else if(_cfg_len <= _old_cfg_status.start){
    //     cfgBaseAddr = 0;
    // }else{ // cfg data space overlap last cfg data space
    //     cfgBaseAddr = 0;
    //     _ld_cfg_dep = LD_DEP_EX_LAST_TASK;
    // }
    // _old_cfg_status.start = cfgBaseAddr;
    // _old_cfg_status.end = cfgBaseAddr + (_cfg_len +  cfgSpadDataByte - 1) / cfgSpadDataByte * cfgSpadDataByte;
    // _old_cfg_status.end = std::min(_old_cfg_status.end, cfgSpadSize);
  int cfg_len = cfgNum * (alignWidth + cfgDataWidth) / 8; // length of config_addr and config_data in bytes
  int cfgBaseAddr = 0;
  int banks = adg->numIobNodes();
  int sizeofBank = adg->iobSpadBankSize();
  int cfgBaseAddrSpad = cfgBaseAddr + banks * sizeofBank; // cfg spad on top of iob spad
  int cfgBaseAddrCtrl = cfgBaseAddr / cfgSpadDataByte; // config base address the controller access
  
  BYTES_LIST iob_ens = _kernel_to_iob_ens[kernel];

  CFGandEXE << "data_ptr.append(iptrs)\n";
  // CFGandEXE << "data_ptr.append(optrs)\n\n";

  CFGandEXE << "config_" << kernel.getKernelName() << "= DeviceConfig(\n" ;
  CFGandEXE << "\tconfig_values=" << CFGarrayName << ",\n" ;
  CFGandEXE << "\tiob_en=[" ;
  for(int _ = 0; _ < iob_ens.size(); _++){
    CFGandEXE << iob_ens.getByte(_);
    if(_ != iob_ens.size() - 1)
      CFGandEXE << "," ;
  }
  CFGandEXE << "],\n" ;
  CFGandEXE << "\tdata_ptr=data_ptr\n";
  CFGandEXE << ")\n" ;

  CFGandEXE << "configs.append(config_" << kernel.getKernelName() << ")";

  // CFGandEXE << "\tload_cfg((void*)" << CFGarrayName << ", 0x" << std::hex << cfgBaseAddrSpad << std::dec << ", " 
  //      << cfg_len << ", " << /*_task_id=*/"_task_id" << ", " << /*_ld_cfg_dep*/"LD_DEP_EX_LAST_TASK" << ");\n";
  // CFGandEXE << "config(0x" << std::hex << cfgBaseAddrCtrl << std::dec << ", " << cfgNum << ", " << /*_task_id*/"_task_id" << ", " << /*_ex_dep*/ 0 << ");\n";
  // CFGandEXE << "execute(0x" << std::hex << iob_ens << std::dec << ", " << /*_task_id*/"_task_id" << ", " << /*_ex_dep*/"EX_DEP_ST_LAST_TASK" << ");\n";

  KnToCfgExe[kernel] = CFGandEXE.str();
  
  // std::cout << CFGandEXE.str() << std::endl;
}


/// @brief Get the config and execution instructions of CGRA
/// @param mapper The SA mapper which has completed mapping
void PytestEmitter::GenerateCGRACFGAndEXE(ADORA::KernelOp& kernel, MapperSA* mapper){
  /// why MapperSA (without&) will cause bug ?? memory leakage?
  /// Generate CGRA configuration
  ADG* adg = mapper->getADG();
  Configuration cfg(mapper->_mapping);
  KnToConfiguration[kernel] = cfg;
  _adg = adg;
  GenerateCGRACFGAndEXE(kernel, cfg, adg);
}

/////////////////////////
/// emit Gemm block
/////////////////////////
/// @brief Emit a block nested in "for" op for adora Gemm
/// @param os
void PytestEmitter::emitGemmBlock(mlir::Block &block, llvm::raw_ostream &os) {
  // std::stringstream ostr;
  addIndent();

  opEmitter->setIndent(getIndent());
  block.dump();

  for (auto &op : block) {
    op.dump();
    // TypeSwitch<Operation *, bool>(&op)
    //   .template Case<
    //     // Affine statements.
    //     affine::AffineForOp,
    //     // // Special expressions.
    //     arith::ConstantOp
    //   >([&](auto opNode) -> bool {
    //     op.dump();
    //     return true;
    //   })
    //   .Default([&](auto opNode) -> bool {
    //     llvm::errs() << "No support!\n";
    //     return false;
    //   });
    if(opEmitter->dispatchVisitor(&op)){
      continue;
    }
    else{
      op.emitError("can't be correctly emitted.");
    } 
  }
  reduceIndent();
  opEmitter->setIndent(getIndent());
  // os << ostr.str();
}