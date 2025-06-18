//===----------------------------------------------------------------------===//
//
// Copyright 2022-2023 The CGRVOPT Authors.
//
//===----------------------------------------------------------------------===//
#include "emit/EmitCGRACall.h"
#include "emit/OpVisitor.h"
#include "mlir/Dialect/Affine/Utils.h"

using namespace mlir;
using namespace mlir::ADORA;


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

/// @brief Get a new id for new op which will be emit to C 
/// @param value_name_list
int NewValueNameId(const llvm::SmallDenseMap<mlir::Value, Op_Name_C>& value_name_list){
  int new_id = 0;
  for(auto& elem: value_name_list){
      new_id = new_id < elem.second.id ? elem.second.id : new_id;
  }
  return new_id + 1;
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
  });
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
class CGRVOpEmitter : public MLIROpVisitorBase<CGRVOpEmitter, bool> {
public:
  CGRVOpEmitter(llvm::raw_ostream &os) : _os(os) {}
  CGRVOpEmitter(CGRACallEmitter& emitter, llvm::raw_ostream &os) : 
      _cgracallemitter(&emitter) ,_os(os) {setIndent(emitter.getIndent());}
  using MLIROpVisitorBase::visitOp;

  /// Tool functions for emitting
  raw_ostream& indent(){return _os.indent(_indent);}
  void setIndent(unsigned newindent){ _indent = newindent;}

  /// @brief emit a new op to C, add this one to op_name_list. 
  /// @param mlirop the corresponding mlir operation
  /// @param type the C type of this operation
  /// @return 
  std::string EmitNewValueAndGetName(mlir::Value v, const std::string type){
    if(_cgracallemitter->getValueNameList().count(v)){
      return _cgracallemitter->lookupName(v);
    }
    Op_Name_C newopinfo;
    newopinfo.id = NewValueNameId(_cgracallemitter->getValueNameList());
    newopinfo.type = type;
    _cgracallemitter->appendValueNameList(v, newopinfo);
    return newopinfo.name();
  }

  template <typename opT>
    bool EmitBinary(opT op ,std::string op_symbol){
      std::string type = getEmitType(op.getResult());
      std::string Lhs = _cgracallemitter->lookupName(op.getLhs());
      if(Lhs == "")
        Lhs = ConstOpToValueStr[op.getLhs()];
      std::string Rhs = _cgracallemitter->lookupName(op.getRhs());
      if(Rhs == "")
        Rhs = ConstOpToValueStr[op.getRhs()];
      assert(Lhs != "" && Rhs != "");

      indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Lhs << " " << op_symbol << " " << Rhs << ";\n"; 
      return true;
    }

  /// ADORA dialect operations.
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
    indent() << "{\n";
    indent() << "/// " << op << "\n";

    std::string BLid = op.getId().str();
    ::llvm::ArrayRef<int64_t> SourceShape =  op.getOriginalMemrefType().getShape();
    ::llvm::ArrayRef<int64_t> ResultShape =  op.getResultType().getShape();
    // MemRefType SourceType = op.getOriginalMemref();
    // MemRefType ResultType = op.getResultType();
    if(SourceShape.size() == 0){
      //// memref<f32> -> memref<2xf32>
      std::string Memref_BaseAddr = _cgracallemitter->lookupName(op.getOriginalMemref());
      llvm::SmallVector<dfgIoInfo> DfgIoInfos = _cgracallemitter->getDfgIoInfosFromBlockLoad(op);
      assert(DfgIoInfos.size() == 1);
      uint64_t spadbaddr = DfgIoInfos[0].addr;
      uint64_t DMA_Len = 8; /// 64bit
      int fuse = 0;
      std::stringstream load_data;
      load_data <<"load_data(*" << Memref_BaseAddr  ////address not pointer
              <<", 0x" << std::hex << spadbaddr 
              <<", " << std::dec << DMA_Len 
              <<", " << std::dec << fuse  /*fuse*/
              <<", _task_id" /*Task id*/ << ", LD_DEP_ST_LAST_TASK" /*Task dep*/
              <<");\n";
      indent() << load_data.str();
      _os << "\n";
      indent() << "}\n";
      return true;
    }

    assert(SourceShape.size() == ResultShape.size());

    /// Get DMA_Len 
    uint64_t DataBytes = op.getOriginalMemrefType().getElementTypeBitWidth()/8;
    uint64_t DMA_Len = DataBytes;

    for(int r = SourceShape.size() - 1; r >= 0; r--){
      // assert(SourceShape[r] >= ResultShape[r]);
      if(op.getOriginalMemrefType().isDynamicDim(r)){
        DMA_Len = DMA_Len * ResultShape[r];
        break;
      }
      else{
        assert(SourceShape[r] >= ResultShape[r]);
        DMA_Len = DMA_Len * ResultShape[r];
        if(SourceShape[r] > ResultShape[r])
          break;
      }
    }
    
    /// Get DRAM_BaseAddr
    std::string Memref_BaseAddr = _cgracallemitter->lookupName(op.getOriginalMemref());

    /// Get DRAM_Offset
    std::string DRAM_Offset = "";
    for(int operandIdx = 0; operandIdx < op.getMapOperands().size(); operandIdx++){
      mlir::Value operand = op.getMapOperands()[operandIdx];
      std::string operandname = _cgracallemitter->lookupName(operand);
      if(operandname == "")      
        operandname = ConstOpToValueStr[operand];
      assert(operandname != "");
      if(operandname == "0")
        continue;
      else{
        SmallVector<int>Dimensions = getOperandDimensionsInMap(/*dim=*/operandIdx, /*map=*/op.getAffineMap());
        // assert(Dimensions.size() == 1);

        for(unsigned d = 0; d < Dimensions.size(); d++){
          int64_t elements_each_step = DataBytes;
          for (unsigned i = Dimensions[d] + 1; i < SourceShape.size(); i++){
            elements_each_step *= SourceShape[i];
          }
          if(DRAM_Offset != "")
            DRAM_Offset = DRAM_Offset + " + ";
          DRAM_Offset = DRAM_Offset + std::to_string(elements_each_step) + " * " + operandname;
        }
      }
    }
    if(DRAM_Offset == "") {
      if(op.getAffineMap().isEmpty()){
        ///// For %2 = ADORA.BlockLoad %arg2 [] : memref<?xi32> -> memref<2xi32>
        DRAM_Offset = "0";
      }
      else{
        ///// For %2 = ADORA.BlockLoad %arg2 [11, 10] : memref<20x506xi32> -> memref<2x506xi32>
        for(int exprIdx = 0; exprIdx < op.getAffineMap().getResults().size(); exprIdx++){
          AffineExpr expr = op.getAffineMap().getResult(exprIdx);
          assert(expr.getKind() == AffineExprKind::Constant);
          std::string cstValue = std::to_string(expr.dyn_cast<AffineConstantExpr>().getValue());
          assert(cstValue != "");
          if(cstValue == "0")
            continue;
          else{
            // SmallVector<int>Dimensions = getOperandDimensionsInMap(/*dim=*/exprIdx, /*map=*/op.getAffineMap());
            int64_t elements_each_step = DataBytes;
            for (unsigned i = exprIdx + 1; i < SourceShape.size(); i++){
              elements_each_step *= SourceShape[i];
            }
            if(DRAM_Offset != "")
              DRAM_Offset = DRAM_Offset + " + ";
            DRAM_Offset = DRAM_Offset + std::to_string(elements_each_step) + " * " + cstValue;
          }
        }
      }
      if(DRAM_Offset == "") {
        DRAM_Offset = "0";
      }
    }

    /// Get DMA_Request_Offset
    std::vector<int64_t> DMA_Request_Offsets;
    bool continuous = true;
    for(int r = SourceShape.size() - 1; r >= 0; r--){
      if(!continuous){
        if(SourceShape[r] == 1)
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), -1); /// -1 means transfer of this rank is overlooked
        else 
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), ResultShape[r]);
      }
      else {/// continuous
        if(SourceShape[r] > ResultShape[r]){
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), -1); /// -1 means transfer of this rank is continuous
          continuous = false;
        }
        else if(SourceShape[r] == ResultShape[r]){
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), -1); /// -1 means transfer of this rank is continuous
        }
      }
    }    

    /// SPAD_BaseAddr
    /// SPAD_Offset: keep increasing by DMA_Len.
    llvm::SmallVector<dfgIoInfo> DfgIoInfos = _cgracallemitter->getDfgIoInfosFromBlockLoad(op);
    llvm::SmallVector<uint64_t> SPAD_BaseAddrs;
    for(dfgIoInfo elem: DfgIoInfos){
      SPAD_BaseAddrs.push_back(elem.addr);
    }

    /// Emit C
    unsigned cur_indent = _indent;
    unsigned idx_num = 0;
    // llvm::SmallDenseMap<int, int64_t> LItoStep;
    indent() << "uint64_t dramoffset_" << BLid << " = " << DRAM_Offset <<";\n";
    std::string DMA_Request_Offsets_str = "uint64_t roffset_" + BLid + " = ";
    indent() << "uint64_t spadoffset_" << BLid << " = 0;\n";
    for(int r = 0; r < DMA_Request_Offsets.size(); r++){
      int64_t Roffset = DMA_Request_Offsets[r];
      std::string idx = "idx_" + std::to_string(r);
      if(Roffset != -1){
        idx_num++;
        int64_t elements_each_step = DataBytes;
        for (unsigned i = r + 1; i < SourceShape.size(); i++){
          elements_each_step *= SourceShape[i];
        }
        indent() << "for(int " << idx << " = " << 0 << "; "
              << idx << " < " <<  Roffset << "; "
              << idx << "++){\n";
        // LItoStep[idx] = elements_each_step;
        DMA_Request_Offsets_str += " " + std::to_string(elements_each_step)
                                + "*" + idx + " +";
        _indent += 2;
        setIndent(_indent);
      }
    }
    if(DMA_Request_Offsets_str.substr(DMA_Request_Offsets_str.size()-2, 2) == "= ")
      DMA_Request_Offsets_str += "0;";
    else
      DMA_Request_Offsets_str.back() = ';';
    indent() << DMA_Request_Offsets_str << "\n";
    // _indent += 2;
    // setIndent(_indent);
    // for(auto spadbaddr : SPAD_BaseAddrs){
    for(int i = 0; i < SPAD_BaseAddrs.size(); i++)
    {
      auto spadbaddr = SPAD_BaseAddrs[i];
      int fuse = (i == SPAD_BaseAddrs.size() - 1)? 0 : 1; /// fuse: 0 - broadcast, 1 - non-broadcast
      std::stringstream load_data;
      load_data <<"load_data(" << Memref_BaseAddr 
              <<" + " << "dramoffset_" << BLid
              <<" + " << "roffset_" << BLid
              <<", 0x" << std::hex << spadbaddr 
              <<" + " << "spadoffset_" << BLid 
              <<", " << std::dec << DMA_Len 
              <<", " << std::dec << fuse  /*fuse*/
              <<", _task_id" /*Task id*/ << ", LD_DEP_ST_LAST_TASK" /*Task dep*/
              <<");\n";
      indent() << load_data.str();

    }
    indent()<< "spadoffset_" << BLid 
            << " = spadoffset_" << BLid 
            << " + " << DMA_Len <<";\n";



    _indent = cur_indent;
    setIndent(_indent);
    indent();
    for(int _ = 0 ; _ < idx_num ; _++){
      _os << "} ";
    }
    _os << "\n";
    indent() << "}\n";

    return true;
  }

  //////
  /// Check whether a DataBlockStoreOp is the last of one kernel in one Block.
  ///
  static bool IsLastBlockStoreOp(ADORA::DataBlockStoreOp op){
    Block* parentBlock = op.getOperation()->getBlock();

    bool behindThisOp = false;
    for(auto it = parentBlock->begin(); it != parentBlock->end(); it++){
      if(behindThisOp && isa<ADORA::DataBlockStoreOp>(it)){
        ADORA::DataBlockStoreOp toCheck = dyn_cast<ADORA::DataBlockStoreOp>(it);
        
        if(toCheck.getKernelName() == op.getKernelName()){
          //// this is not the last datablockop of one kernel in this block
          return false;
        }
      }
      if(&*it == op.getOperation()){
        behindThisOp = true;
      }
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
    indent() << "{\n";
    indent() << "/// " << op << "\n";

    std::string BLid = op.getId().str();
    ::llvm::ArrayRef<int64_t> SourceShape =  op.getSourceMemrefType().getShape();
    ::llvm::ArrayRef<int64_t> TargetShape =  op.getTargetMemrefType().getShape();
    // MemRefType SourceType = op.getOriginalMemref();
    // MemRefType ResultType = op.getResultType();
    if(TargetShape.size() == 0){
      //// memref<2xf32> -> memref<f32> 
      std::string Memref_BaseAddr = _cgracallemitter->lookupName(op.getTargetMemref());
      // llvm::SmallVector<dfgIoInfo> DfgIoInfos = _cgracallemitter->getDfgIoInfosFromBlockLoad(op);
      dfgIoInfo DfgIoInfo = _cgracallemitter->getDfgIoInfosFromBlockStore(op);
      // assert(DfgIoInfos.size() == 1);
      uint64_t spadbaddr = DfgIoInfo.addr;
      uint64_t DMA_Len = 8; /// 64bit
      int fuse = 0;

      std::stringstream store_data;
      store_data <<"store(&" << Memref_BaseAddr  //// address not pointer
              <<", 0x" << std::hex << spadbaddr 
              <<", " << std::dec << DMA_Len 
              <<", _task_id" /*Task id*/ << ", 0" /*Task dep*/
              <<");\n";
      indent() << store_data.str();

      _os << "\n";
      indent() << "}\n";

      if(IsLastBlockStoreOp(op)){
        indent() << "_task_id++;\n";
      }

      return true;
    }

    assert(SourceShape.size() == TargetShape.size());

    /// Get DMA_Len 
    uint64_t DataBytes = op.getSourceMemrefType().getElementTypeBitWidth()/8;
    uint64_t DMA_Len = DataBytes;

    for(int r = SourceShape.size() - 1; r >= 0; r--){
      if(op.getTargetMemrefType().isDynamicDim(r)){
        DMA_Len = DMA_Len * SourceShape[r];
        break;
      }
      else{
        assert(SourceShape[r] <= TargetShape[r]);
        DMA_Len = DMA_Len * SourceShape[r];
        if(SourceShape[r] < TargetShape[r])
          break;
      }
    }
    
    /// Get DRAM_BaseAddr
    std::string Memref_BaseAddr = _cgracallemitter->lookupName(op.getTargetMemref());

    /// Get DRAM_Offset
    std::string DRAM_Offset = "";
    for(int operandIdx = 0; operandIdx < op.getMapOperands().size(); operandIdx++){
      mlir::Value operand = op.getMapOperands()[operandIdx];
      operand.dump();
      std::string operandname = _cgracallemitter->lookupName(operand);
      std::cout << operandname <<"\n";
      if(operandname == "")      
        operandname = ConstOpToValueStr[operand];
      assert(operandname != "");
      if(operandname == "0")
        continue;
      else{
        SmallVector<int>Dimensions = getOperandDimensionsInMap(/*dim=*/operandIdx, /*map=*/op.getAffineMap());
        // assert(Dimensions.size() == 1);

        for(unsigned d = 0; d < Dimensions.size(); d++){
          int64_t elements_each_step = DataBytes;
          for (unsigned i = Dimensions[d] + 1; i < TargetShape.size(); i++){
            elements_each_step *= TargetShape[i];
          }
          if(DRAM_Offset != "")
            DRAM_Offset = DRAM_Offset + " + ";
          DRAM_Offset = DRAM_Offset + std::to_string(elements_each_step) + " * " + operandname;
        }
      }
    }
    if(DRAM_Offset == "") {
      if(op.getAffineMap().isEmpty()){
        ///// ADORA.Blockstore %1, %arg2 [] :  memref<2xi32> -> memref<?xi32> 
        DRAM_Offset = "0";
      }
      else{
        ///// ADORA.Blockstore %1, %arg2 [11, 10] : memref<2x506xi32> -> memref<20x506xi32>
        for(int exprIdx = 0; exprIdx < op.getAffineMap().getResults().size(); exprIdx++){
          AffineExpr expr = op.getAffineMap().getResult(exprIdx);
          assert(expr.getKind() == AffineExprKind::Constant);
          std::string cstValue = std::to_string(expr.dyn_cast<AffineConstantExpr>().getValue());
          assert(cstValue != "");
          if(cstValue == "0")
            continue;
          else{
            // SmallVector<int>Dimensions = getOperandDimensionsInMap(/*dim=*/exprIdx, /*map=*/op.getAffineMap());
            int64_t elements_each_step = DataBytes;
            for (unsigned i = exprIdx + 1; i < TargetShape.size(); i++){
              elements_each_step *= TargetShape[i];
            }
            if(DRAM_Offset != "")
              DRAM_Offset = DRAM_Offset + " + ";
            DRAM_Offset = DRAM_Offset + std::to_string(elements_each_step) + " * " + cstValue;
          }
        }
      }
      if(DRAM_Offset == "") {
        DRAM_Offset = "0";
      }
    }

    /// Get DMA_Request_Offset
    std::vector<int64_t> DMA_Request_Offsets;
    bool continuous = true;
    for(int r = SourceShape.size() - 1; r >= 0; r--){
      if(!continuous){
        if(SourceShape[r] == 1)
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), -1); /// -1 means transfer of this rank is overlooked
        else 
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), SourceShape[r]);
      }
      else {/// continuous
        if(TargetShape[r] > SourceShape[r]){
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), -1); /// -1 means transfer of this rank is continuous
          continuous = false;
        }
        else if(TargetShape[r] == SourceShape[r]){
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), -1); /// -1 means transfer of this rank is continuous
        }
      }
    }    

    /// SPAD_BaseAddr
    /// SPAD_Offset: keep increasing by DMA_Len.
    dfgIoInfo DfgIoInfos = _cgracallemitter->getDfgIoInfosFromBlockStore(op);
    llvm::SmallVector<uint64_t> SPAD_BaseAddrs;
    SPAD_BaseAddrs.push_back(DfgIoInfos.addr);

    /// Emit C
    unsigned cur_indent = _indent;
    unsigned idx_num = 0;
    // llvm::SmallDenseMap<int, int64_t> LItoStep;
    indent() << "uint64_t dramoffset_" << BLid << " = " << DRAM_Offset <<";\n";
    std::string DMA_Request_Offsets_str = "uint64_t roffset_" + BLid + " = ";
    indent() << "uint64_t spadoffset_" << BLid << " = 0;\n";
    for(int r = 0; r < DMA_Request_Offsets.size(); r++){
      int64_t Roffset = DMA_Request_Offsets[r];
      std::string idx = "idx_" + std::to_string(r);
      if(Roffset != -1){
        idx_num++;
        int64_t elements_each_step = DataBytes;
        for (unsigned i = r + 1; i < TargetShape.size(); i++){
          elements_each_step *= TargetShape[i];
        }
        indent() << "for(int " << idx << " = " << 0 << "; "
              << idx << " < " <<  Roffset << "; "
              << idx << "++){\n";
        // LItoStep[idx] = elements_each_step;
        DMA_Request_Offsets_str += " " + std::to_string(elements_each_step)
                                + "*" + idx + " +";
        _indent += 2;
        setIndent(_indent);
      }
    }
    if(DMA_Request_Offsets_str.substr(DMA_Request_Offsets_str.size()-2, 2) == "= ")
      DMA_Request_Offsets_str += "0;";
    else
      DMA_Request_Offsets_str.back() = ';';
    indent() << DMA_Request_Offsets_str << "\n";
    // _indent += 2;
    // setIndent(_indent);
    // for(auto spadbaddr : SPAD_BaseAddrs){
    for(int i = 0; i < SPAD_BaseAddrs.size(); i++)
    {
      auto spadbaddr = SPAD_BaseAddrs[i];
      int fuse = (i == SPAD_BaseAddrs.size() - 1)? 0 : 1; /// fuse: 0 - broadcast, 1 - non-broadcast
      std::stringstream store_data;
      store_data <<"store(" << Memref_BaseAddr 
              <<" + " << "dramoffset_" << BLid
              <<" + " << "roffset_" << BLid
              <<", 0x" << std::hex << spadbaddr 
              <<" + " << "spadoffset_" << BLid 
              <<", " << std::dec << DMA_Len 
              <<", _task_id" /*Task id*/ << ", 0" /*Task dep*/
              <<");\n";
      indent() << store_data.str();

    }
    indent()<< "spadoffset_" << BLid 
            << " = spadoffset_" << BLid 
            << " + " << DMA_Len <<";\n";


    _indent = cur_indent;
    setIndent(_indent);
    indent();
    for(int _ = 0 ; _ < idx_num ; _++){
      _os << "} ";
    }
    _os << "\n";
    indent() << "}\n";

    if(IsLastBlockStoreOp(op)){
      indent() << "_task_id++;\n";
    }

    return true;    
  }

  bool visitOp(ADORA::LocalMemAllocOp op) {
    
    return true;
  }
  


  bool visitOp(ADORA::KernelOp op) {
    indent() << "{\n";
    if(!MapHasKey(_cgracallemitter->KnToCfgExe, op)){
      // Configuration cfg = ;
      ADG* adg = _cgracallemitter->getADG();
      _cgracallemitter->GenerateCGRACFGAndEXE(op, _cgracallemitter->KnToConfiguration[op], adg);
    }
    if(!op.getKernelName().empty()){
      indent() << "/// " << op.getKernelName() << "\n";
    }

    std::vector<std::string> strs = split_str_by_char(_cgracallemitter->KnToCfgExe[op], '\n');
    for(std::string str: strs){
      indent() << str << "\n";
    }
    
    indent() << "}\n";
    return true;
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
    indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) << ";\n";

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
    _cgracallemitter->emitBlock(*(op.getDest()), _os);
    return true;
  }

  /// Affine statements.
  bool visitOp(affine::AffineForOp op) { 
    indent() << "for (";
    auto iterVar = op.getInductionVar();

    // Emit lower bound.
    assert(op.getLowerBoundMap().getResults().size()==1);
    _os << "int " << EmitNewValueAndGetName(iterVar, "int") << " = ";
    _os << op.getLowerBoundMap().getResult(0) << "; ";

    // Emit loop invariant(upper bound)
    assert(op.getUpperBoundMap().getResults().size()==1);
    _os << _cgracallemitter->lookupName(iterVar) << " < " ;
    _os << op.getUpperBoundMap().getResult(0) << "; ";

    // Emit loop step
    _os << _cgracallemitter->lookupName(iterVar) << " = " ;
    _os << _cgracallemitter->lookupName(iterVar) << " + "  << op.getStep() << "){\n";

    _cgracallemitter->emitBlock(*(op.getBody()), _os);
    // reduce
    indent() << "}\n";
    indent() << "\n";
    indent() << "\n";
    indent() << "\n";
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
      std::string value = _cgracallemitter->lookupName(op.getValue());
      // assert("Unsupported!\n");
    }

    assert(op.getMemref().getType().cast<MemRefType>().getShape().size() == 0);
    std::string memref = _cgracallemitter->lookupName(op.getMemref());
    indent() << memref << " = " << value << ";\n";
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
      std::string Lhs = _cgracallemitter->lookupName(op.getLhs());
      if(Lhs == "")
        Lhs = ConstOpToValueStr[op.getLhs()];
      std::string Rhs = _cgracallemitter->lookupName(op.getRhs());
      if(Rhs == "")
        Rhs = ConstOpToValueStr[op.getRhs()];
      assert(Lhs != "" && Rhs != "");

      if( Lhs == "0" && Rhs == "0")
        indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << "0" << ";\n";
      else if( Lhs == "0" && Rhs != "0")
        indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Rhs << ";\n";
      else if( Lhs != "0" && Rhs == "0")
        indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Lhs << ";\n";
      else
        indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Lhs << " " << "+" << " " << Rhs << ";\n"; 
      
      return true;
    // return EmitBinary(op, "+");
  }

  bool visitOp(arith::SubIOp op) {
      std::string type = getEmitType(op.getResult());
      std::string Lhs = _cgracallemitter->lookupName(op.getLhs());
      if(Lhs == "")
        Lhs = ConstOpToValueStr[op.getLhs()];
      std::string Rhs = _cgracallemitter->lookupName(op.getRhs());
      if(Rhs == "")
        Rhs = ConstOpToValueStr[op.getRhs()];
      assert(Lhs != "" && Rhs != "");

      if( Lhs == "0" && Rhs == "0")
        indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << "0" << ";\n";
      else if( Lhs != "0" && Rhs == "0")
        indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Lhs << ";\n";
      else
        indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
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
  CGRACallEmitter* _cgracallemitter;
  llvm::DenseMap<mlir::Value, std::string> ConstOpToValueStr;
  llvm::raw_ostream &_os;
  unsigned _indent = 0;
};
CGRVOpEmitter* opEmitter;
} // namespace


// static InFlightDiagnostic emitError(Operation *op, const Twine &message) {
//   // state.encounteredError = true;
//   return op->emitError(message);
// }
//===----------------------------------------------------------------------===//
// Members of CGRACallEmitter class
//===----------------------------------------------------------------------===//
/// @brief Emit the header of a function including function name, function args...
/// @param os
void CGRACallEmitter::emitFunctionHead(func::FuncOp &funcop, llvm::raw_ostream &os) {
  std::stringstream ostr;
  ostr << "void " << funcop.getSymName().str() << "(";
  // Funtion args
  ArrayRef<mlir::Type> argTypes = funcop.getArgumentTypes();
  for(int argIdx = 0; argIdx < argTypes.size(); argIdx++){
    if(argIdx != 0){
      ostr << " ,";
    }
    mlir::Type argType = argTypes[argIdx];

    // argType.dump();
    Op_Name_C arg_info("arg", argIdx);
    appendValueNameList(funcop.getBody().getArgument(argIdx), arg_info);
    if(argType.isa<MemRefType>()){
      ostr << "void* " << "arg_" << argIdx;
    } 
    else if(argType.isIndex()){
      ostr << "int " << "arg_" << argIdx;
    }
    else if(argType.isInteger(32)){
      ostr << "int32_t " << "arg_" << argIdx;
    }
    else if(argType.isInteger(64)){
      ostr << "int64_t " << "arg_" << argIdx;
    }
    else if(argType.isUnsignedInteger(32)){
      ostr << "uint32_t " << "arg_" << argIdx;
    }
    else if(argType.isUnsignedInteger(64)){
      ostr << "uint64_t " << "arg_" << argIdx;
    }
    
  }
  ostr << "){\n";


  os << ostr.str();
}

/// @brief Emit a block (maybe a loop body, maybe a function body), especially the for loop structure
/// @param os
void CGRACallEmitter::emitBlock(mlir::Block &block, llvm::raw_ostream &os) {
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
bool CGRACallEmitter::emitCGRACallFunction(llvm::raw_ostream &os) {
  opEmitter = new CGRVOpEmitter(*this, os);
  // ADORAEmitterState state(os);
  // ModuleEmitter(state).emitModule(module);
  // return failure(state.encounteredError);
  os << R"XXX(
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

uint8_t _task_id = 0;

#define LD_DEP_ST_LAST_TASK 1     // this load command depends on the store command of last task
#define LD_DEP_EX_LAST_TASK 2     // this load command depends on the execute command of last task
#define LD_DEP_ST_LAST_SEC_TASK 3 // this load command depends on the store command of last second task
#define EX_DEP_ST_LAST_TASK 1     // this EXECUTE command depends on the store command of last task


)XXX";

  //// emit configuration data array
  os << R"XXX(
//===----------------------------------------------------------------------===//
// Configuration Data 
//===----------------------------------------------------------------------===//
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
    os << "}\n";
  }

  delete opEmitter;
}

/// @brief Get SPAD information (which bank to transfer data, data size...) for every data block load.
///        This information is store in _LoadToDfgIoInfos/_StoreToDfgIoInfo
/// @param kernel Kernel which has been translated to CDFG
/// @param mapper The SA mapper which has completed mapping
void CGRACallEmitter::DataBlockOperationsToSPADInfo(ADORA::KernelOp& kernel, MapperSA* mapper){
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
  /// Get io information of every block load ops, including Addr(Spad), iobAddr, LorS
  _moduleop.walk([&](ADORA::DataBlockLoadOp blockload) {  
    if(blockload.getKernelName() == kernel.getKernelName()){
      std::string BlockLoadName = blockload.getKernelName().str() + "_" + blockload.getId().str();

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
        }
      }
    }
  });


  /// Get io information of every block store ops, including Addr(Spad), iobAddr, LorS
  _moduleop.walk([&](ADORA::DataBlockStoreOp blockstore) {  
    if(blockstore.getKernelName() == kernel.getKernelName()){
      std::string BlockStoreName = blockstore.getKernelName().str() + "_" + blockstore.getId().str();

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
        }
      }
    }
  });

  _kernel_to_dfg_io_infos[kernel] = dfg_io_infos;
  _kernel_to_iob_ens[kernel] = iob_ens;
}

// Traverse the whole module to find which operation keeps a "VAR_CONFIG" attr
// equal to arg config, and return the result value of the operation
std::string CGRACallEmitter::lookupVarConfigName(const std::string config){ 
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
  });
  std::string target_name = lookupName(target_value);
  assert(target_name != "");
  return target_name;
}

void CGRACallEmitter::setMapResult(ADORA::KernelOp k, MapperSA* mapper){
  _adg = mapper->getADG();
  KnToConfiguration[k] = Configuration(mapper->_mapping);
}

/// @brief Get the config data
std::string CGRACallEmitter::GenerateCGRAConfig(
  ADORA::KernelOp& kernel, Configuration cfg, ADG* adg){
  // adg->print();
  std::string CFGarrayName = "cin_" + kernel.getKernelName();
  std::stringstream CFGdata;
  CFGdata << "/// " << kernel.getKernelName() << "\n";

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

  if(cfgAddrWidth > 16){
    CFGdata << "volatile unsigned int ";
  }else{
    CFGdata << "volatile unsigned short ";
  }
  CFGdata << CFGarrayName << "[" << cfgNum << "][" << (1 + cfgDataWidth / alignWidth) << "] __attribute__((aligned(" << cfgSpadDataByte << "))) = {\n";
  CFGdata << std::hex;
  int alignWidthHex = alignWidth/4;
  for(auto& cdp : cfgData){
    CFGdata << "\t\t{";
    for(auto data : cdp.data){      
      if(alignWidth == 32){
         CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << data << ", ";
      }else{
        CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (data & 0xffff) << ", ";
        CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (data >> 16) << ", ";
      }
            
    }
    CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (cdp.addr) << "},\n";
  }
  CFGdata << std::dec << "\t};\n\n";

  KnToCfgData[kernel] = CFGdata.str();
  KnToCfgArrayInfo[kernel] = std::pair(CFGarrayName, cfgNum);

  return CFGdata.str();
}

/// @brief Get the config data
std::string CGRACallEmitter::GenerateCGRAConfig(
  ADORA::KernelOp& kernel, MapperSA* mapper){
  ADG* adg = mapper->getADG();
  Configuration cfg(mapper->_mapping);
  KnToConfiguration[kernel] = cfg;
  _adg = adg;
  return GenerateCGRAConfig(kernel, cfg, adg);
}



/// @brief Get the config and execution instructions of CGRA
void CGRACallEmitter::GenerateCGRACFGAndEXE(
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
                  << std::hex << " & 0x" << replace.getMask() << ");\n" ;
      }
      else{
        // left shift
        CFGandEXE << std::hex << "(" << varcfg_name << " << 0x" << replace.lshift << ")"
                  << " | " 
                  << std::dec << "(" << CFGarrayName <<"[" << replace.Idx0 << "][" << replace.Idx1 << "]" 
                  << std::hex << " & 0x" << replace.getMask() << ");\n" ;
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
  
  uint64_t iob_ens = _kernel_to_iob_ens[kernel];

  CFGandEXE << "load_cfg((void*)" << CFGarrayName << ", 0x" << std::hex << cfgBaseAddrSpad << std::dec << ", " 
       << cfg_len << ", " << /*_task_id=*/"_task_id" << ", " << /*_ld_cfg_dep*/"LD_DEP_EX_LAST_TASK" << ");\n";
  CFGandEXE << "config(0x" << std::hex << cfgBaseAddrCtrl << std::dec << ", " << cfgNum << ", " << /*_task_id*/"_task_id" << ", " << /*_ex_dep*/ 0 << ");\n";
  CFGandEXE << "execute(0x" << std::hex << iob_ens << std::dec << ", " << /*_task_id*/"_task_id" << ", " << /*_ex_dep*/"EX_DEP_ST_LAST_TASK" << ");\n";

  KnToCfgExe[kernel] = CFGandEXE.str();
  
  // std::cout << CFGandEXE.str() << std::endl;
}


/// @brief Get the config and execution instructions of CGRA
/// @param mapper The SA mapper which has completed mapping
void CGRACallEmitter::GenerateCGRACFGAndEXE(ADORA::KernelOp& kernel, MapperSA* mapper){
  /// why MapperSA (without&) will cause bug ?? memory leakage?
  /// Generate CGRA configuration
  ADG* adg = mapper->getADG();
  Configuration cfg(mapper->_mapping);
  KnToConfiguration[kernel] = cfg;
  _adg = adg;
  GenerateCGRACFGAndEXE(kernel, cfg, adg);
}

