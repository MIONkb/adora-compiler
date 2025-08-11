//===- ADORAOps.cpp - Operations of the ADORA dialect -------------------------===//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Affine/IR/AffineValueMap.h"
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "RAAA/Dialect/ADORA/IR/ADORA.h"

#include "RAAA/Dialect/ADORA/IR/ADORAOps.h.inc"


using namespace mlir;
using namespace mlir::ADORA;
using namespace mlir::func;
using namespace mlir::affine;

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//
void KernelOp::build(OpBuilder &builder, OperationState &result) {

  // Add the data operands.

  // This is a good area to add static operands.

  // Create a kernel body region with kNumConfigRegionAttributes + N arguments,
  // where the first kNumConfigRegionAttributes arguments have `index` type and
  // the rest have the same types as the data operands.
  Region *kernelRegion = result.addRegion();
  Block *body = new Block();
  for (unsigned i = 0; i < kNumConfigRegionAttributes; ++i)
    body->addArgument(builder.getIndexType(), result.location);
  kernelRegion->push_back(body);
}

void KernelOp::build(OpBuilder &builder, OperationState &result, std::string KernelName) {
  auto KernelNameAttr = builder.getStringAttr(KernelName);
  result.addAttribute(getKernelNameAttrStr(), KernelNameAttr);
  build(builder, result);
}

// void KernelOp::getCanonicalizationPatterns(RewritePatternSet &results,
//                                           MLIRContext *context) {
// }
LogicalResult KernelOp::verify() {
  //  Include this code if Kernel launch takes KNumConfigOperands leading
  //  operands for grid/block sizes and transforms them into
  //  kNumConfigRegionAttributes region arguments for block/thread identifiers
  //  and grid/block sizes.
  // if (!op.body().empty()) {
  //   if (op.body().getNumArguments() !=
  //       LaunchOp::kNumConfigOperands + op.getNumOperands())
  //     return op.emitOpError("unexpected number of region arguments");
  // }

  /// To Fix
  for (Block &block : getBody()) {
    if (block.empty())
      continue;
    if (block.back().getNumSuccessors() != 0)
      continue;
  }
  
  Region& knRegion = getBody();
  /// A kernel should only contain 1 region
  /// and this region should only contain 1 block
  if(knRegion.getBlocks().size() != 1)
    return emitOpError(
        "Kernel Region should only get 1 Block.");
  Block& knBlock = knRegion.front();
  /// this kernel Block should only contain 2 Op(a forOp and a terminatorOp)
  /// To fix: maybe not a forOp
  // if(knBlock.getOperations().size() != 2 )
  //   return emitOpError(
  //       "kernel Block should only get 2 Ops.");
  return success();
}

void KernelOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/true,/*printBlockTerminators=*/true);
  printer.printOptionalAttrDict((*this)->getAttrs());
}

// Parses a Launch operation.
// operation ::= `gpu.launch` region attr-dict?
// ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {

  // Region arguments to be created.
  SmallVector<OpAsmParser::UnresolvedOperand, 16> regionArgs(
      KernelOp::kNumConfigRegionAttributes);
  MutableArrayRef<OpAsmParser::UnresolvedOperand> regionArgsRef(regionArgs);

  // Introduce the body region and parse it. The region has
  // kNumConfigRegionAttributes arguments that correspond to
  // block/thread identifiers and grid/block sizes, all of the `index` type.
  Type index = parser.getBuilder().getIndexType();
  SmallVector<Type, KernelOp::kNumConfigRegionAttributes> dataTypes(
      KernelOp::kNumConfigRegionAttributes, index);

  SmallVector<OpAsmParser::Argument> regionArguments;
  for (auto ssaValueAndType : llvm::zip(regionArgs, dataTypes)) {
    OpAsmParser::Argument arg;
    arg.ssaName = std::get<0>(ssaValueAndType);
    arg.type = std::get<1>(ssaValueAndType);
    regionArguments.push_back(arg);
  }

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArguments) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  return success();
}


//===----------------------------------------------------------------------===//
// DataBlockLoadOp
//===----------------------------------------------------------------------===//

void DataBlockLoadOp::build(OpBuilder &builder, OperationState &result,
                            Value OriginalMemref, AffineMap map, ValueRange mapOperands, 
                            MemRefType resultType, std::string KernelName) {
  assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
  result.addOperands(OriginalMemref);
  result.addOperands(mapOperands);

  result.addAttribute(getMapAttrStr(), AffineMapAttr::get(map));

  auto KernelNameAttr = builder.getStringAttr(KernelName);
  result.addAttribute(getKernelNameAttrStr(), KernelNameAttr);

  result.types.push_back(resultType);
}

void DataBlockLoadOp::build(OpBuilder &builder, OperationState &result,
                            Value OriginalMemref, AffineMap map, ValueRange mapOperands, 
                            MemRefType resultType) {
  assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
  std::string KernelName = ""/*"UnknownKernel"*/;
  build(builder, result, OriginalMemref, map, mapOperands, resultType, KernelName);
}

// static bool addKernelNameAttrInParse
//               (OperationState &result, Builder &builder, const std::string KernelName){
//   result.addAttribute(DataBlockLoadOp::getKernelNameAttrStr(), builder.getStringAttr(KernelName));
//   return true;
// }

ParseResult DataBlockLoadOp::parse(OpAsmParser &parser, OperationState &result) {
  /// example:
  /// %0 = ADORA.BlockLoad %arg1 [%arg9, 0] : memref<32x32xf32> -> memref<1x32xf32> {three_mm_32_kernel_0}
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();

  MemRefType memrefType, resultType;
  OpAsmParser::UnresolvedOperand memrefInfo;
  AffineMapAttr mapAttr;
  StringAttr KernelNameAttr;
  // std::string* KernelName = nullptr;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> mapOperands;

  return failure(
      parser.parseOperand(memrefInfo) ||
      parser.parseAffineMapOfSSAIds(mapOperands, mapAttr,
                                    getMapAttrStr(),
                                    result.attributes) ||
      parser.parseColon() || parser.parseType(memrefType) ||
      parser.parseArrow() || parser.parseType(resultType) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.resolveOperand(memrefInfo, memrefType, result.operands) ||
      parser.resolveOperands(mapOperands, indexTy, result.operands) ||
      parser.addTypeToList(resultType, result.types)
      // parser.parseLBrace() || 
      // parser.parseAttribute(KernelNameAttr, builder.getNoneType(), "KernelName", result.attributes)||
      // parser.parseOptionalKeywordOrString(KernelName) || 
      // addKernelNameAttrInParse(result, builder, *KernelName) ||
      // parser.parseOptionalRBrace()
  );
}

void DataBlockLoadOp::print(OpAsmPrinter &p) {
  p << " " << getOriginalMemref() << " [";
  if (AffineMapAttr mapAttr =
          (*this)->getAttrOfType<AffineMapAttr>(getMapAttrStr()))
    p.printAffineMapOfSSAIds(mapAttr, getMapOperands());
  p << "]";
  p << " : " << getOriginalMemrefType() ;
  p << " -> " << getResultType() << " ";
  // p << "{\""  << getKernelName() << "\"}";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{getMapAttrStr()});
  
}

// Returns true if 'value' is a valid index to an affine operation (e.g.
// affine.load, affine.store, affine.dma_start, affine.dma_wait) where
// `region` provides the polyhedral symbol scope. Returns false otherwise.
static bool isValidAffineIndexOperand(Value value, Region *region) {
  return isValidDim(value, region) || isValidSymbol(value, region);
}

/// Verify common indexing invariants of affine.load, affine.store,
/// affine.vector_load and affine.vector_store.
static LogicalResult
verifyMemoryOpIndexing(Operation *op, AffineMapAttr mapAttr,
                       Operation::operand_range mapOperands,
                       MemRefType memrefType, unsigned numIndexOperands) {
  if (mapAttr) {
    AffineMap map = mapAttr.getValue();
    if (map.getNumResults() != memrefType.getRank())
      return op->emitOpError("affine map num results must equal memref rank");
    if (map.getNumInputs() != numIndexOperands)
      return op->emitOpError("expects as many subscripts as affine map inputs");
  } else {
    if (memrefType.getRank() != numIndexOperands)
      return op->emitOpError(
          "expects the number of subscripts to be equal to memref rank");
  }

  Region *scope = getAffineScope(op);
  for (auto idx : mapOperands) {
    if (!idx.getType().isIndex())
      return op->emitOpError("index to load must have 'index' type");
    // if (!isValidAffineIndexOperand(idx, scope))
    //   return op->emitOpError("index must be a dimension or symbol identifier");
  }

  return success();
}

LogicalResult DataBlockLoadOp::verify() {
  MemRefType memrefType = getOriginalMemrefType();
  if (failed(verifyMemoryOpIndexing(
          getOperation(),
          (*this)->getAttrOfType<AffineMapAttr>(getMapAttrStr()),
          getMapOperands(), memrefType,
          /*numIndexOperands=*/getNumOperands() - 1)))
    return failure();

  if (getOriginalMemrefType().getElementType() != getResultType().getElementType())
    return emitOpError(
        "requires 2 memref types of the same elemental type");

  return success();
}

SmallVector<std::string> DataBlockLoadOp::getKernelNameAsStrVector() {
  Attribute KnNameAttr = this->getOperation()->getAttr("KernelName");
  if(KnNameAttr == nullptr)
    return SmallVector<std::string>();
  else{
    SmallVector<std::string> tokens;
    std::string wholeattr = KnNameAttr.cast<StringAttr>().strref().str();
    size_t start = 0;
    size_t end = 0;
    while ((end = wholeattr.find('+', start)) != std::string::npos) {
        tokens.push_back(wholeattr.substr(start, end - start));
        start = end + 1; 
    }
    std::string lastToken = wholeattr.substr(start);
    if (!lastToken.empty()) { 
      tokens.push_back(lastToken);
    }
    return tokens;
  }
}    
std::string DataBlockLoadOp::addAnotherKernelName(const std::string& newKernel) {
  std::string original = getKernelName().str();

  if (original.empty()) {
    setKernelName(newKernel);
    return newKernel;
  }
  else if(findElement(getKernelNameAsStrVector(), newKernel) != -1){
    return original;
  }
  else{
    setKernelName(original + "+" + newKernel);
    return original + "+" + newKernel;
  }
}

//===----------------------------------------------------------------------===//
// DataBlockStoreOp
//===----------------------------------------------------------------------===//

void DataBlockStoreOp::build(OpBuilder &builder, OperationState &result,
                            Value SourceMemref, Value TargetMemref, 
                            AffineMap map, ValueRange mapOperands, 
                            std::string KernelName) {
  assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
  result.addOperands(SourceMemref);
  result.addOperands(TargetMemref);

  result.addOperands(mapOperands);
  result.addAttribute(getMapAttrStr(), AffineMapAttr::get(map));

  auto KernelNameAttr = builder.getStringAttr(KernelName);
  result.addAttribute(getKernelNameAttrStr(), KernelNameAttr);
}

void DataBlockStoreOp::build(OpBuilder &builder, OperationState &result,
                            Value SourceMemref, Value TargetMemref, 
                            AffineMap map, ValueRange mapOperands) {
  assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
  std::string KernelName = ""/*"UnknownKernel"*/;
  build(builder, result, SourceMemref, TargetMemref, map, mapOperands, KernelName);
}

ParseResult DataBlockStoreOp::parse(OpAsmParser &parser, OperationState &result) {
  /// example: To fix
  /// %0 = ADORA.BlockLoad %arg1 [%arg9, 0] : memref<32x32xf32> -> memref<1x32xf32> {three_mm_32_kernel_0}
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();

  MemRefType sourceType, targetType;
  OpAsmParser::UnresolvedOperand sourceInfo;
  OpAsmParser::UnresolvedOperand targetInfo;
  AffineMapAttr mapAttr;
  StringAttr KernelNameAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> mapOperands;
  return failure(
      parser.parseOperand(sourceInfo) || parser.parseComma() ||
      parser.parseOperand(targetInfo) ||
      parser.parseAffineMapOfSSAIds(mapOperands, mapAttr,
                                    getMapAttrStr(),
                                    result.attributes) ||
      // parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() || parser.parseType(sourceType) ||
      parser.parseArrow() || parser.parseType(targetType) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.resolveOperand(sourceInfo, sourceType, result.operands) ||
      parser.resolveOperand(targetInfo, targetType, result.operands) ||
      parser.resolveOperands(mapOperands, indexTy, result.operands)  
      // parser.parseOptionalAttrDict(result.attributes) ||
      // parser.parseLBrace() ||
      // parser.parseAttribute(KernelNameAttr, "KernelName", result.attributes)||
      // parser.parseOptionalRBrace()
  ) ;
}


void DataBlockStoreOp::print(OpAsmPrinter &p) {
  p << " " << getSourceMemref() << ",";
  p << " " << getTargetMemref() << " [";
  if (AffineMapAttr mapAttr =
          (*this)->getAttrOfType<AffineMapAttr>(getMapAttrStr()))
    p.printAffineMapOfSSAIds(mapAttr, getMapOperands());
  p << "]";
  p << " : " << getSourceMemrefType() ;
  p << " -> " << getTargetMemrefType() << " ";
  // p << "{\""  << getKernelName() << "\"}";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{getMapAttrStr()});
}


LogicalResult DataBlockStoreOp::verify() {
  MemRefType memrefType = getTargetMemrefType();
  if (failed(verifyMemoryOpIndexing(
          getOperation(),
          (*this)->getAttrOfType<AffineMapAttr>(getMapAttrStr()),
          getMapOperands(), memrefType,
          /*numIndexOperands=*/getNumOperands() - 2)))
    return failure();

  if (getTargetMemrefType().getElementType() != getSourceMemrefType().getElementType())
    return emitOpError(
        "requires source and target memref types of the same elemental type");

  return success();
}

//===----------------------------------------------------------------------===//
// LocalMemAllocOp
//===----------------------------------------------------------------------===//

void LocalMemAllocOp::build(OpBuilder &builder, OperationState &result,
                            MemRefType resultType, std::string KernelName) {
  auto KernelNameAttr = builder.getStringAttr(KernelName);
  result.addAttribute(getKernelNameAttrStr(), KernelNameAttr);

  result.types.push_back(resultType);
}

void LocalMemAllocOp::build(OpBuilder &builder, OperationState &result,
                            MemRefType resultType) {
  std::string KernelName = ""/*"UnknownKernel"*/;
  build(builder, result, resultType, KernelName);
}

// static bool addKernelNameAttrInParse
//               (OperationState &result, Builder &builder, const std::string KernelName){
//   result.addAttribute(LocalMemAllocOp::getKernelNameAttrStr(), builder.getStringAttr(KernelName));
//   return true;
// }

ParseResult LocalMemAllocOp::parse(OpAsmParser &parser, OperationState &result) {
  /// example:
  /// %0 = ADORA.BlockLoad %arg1 [%arg9, 0] : memref<32x32xf32> -> memref<1x32xf32> {three_mm_32_kernel_0}
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();

  MemRefType resultType;
  OpAsmParser::UnresolvedOperand memrefInfo;
  StringAttr KernelNameAttr;

  return failure(
      // parser.parseOperand(memrefInfo) ||
      parser.parseType(resultType) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.addTypeToList(resultType, result.types)
      // parser.parseLBrace() || 
      // parser.parseAttribute(KernelNameAttr, builder.getNoneType(), "KernelName", result.attributes)||
      // parser.parseOptionalKeywordOrString(KernelName) || 
      // addKernelNameAttrInParse(result, builder, *KernelName) ||
      // parser.parseOptionalRBrace()
  );
}

void LocalMemAllocOp::print(OpAsmPrinter &p) {
  p << " " << getResultType() << " ";
  // p << "{\""  << getKernelName() << "\"}";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{getMapAttrStr()});
}

LogicalResult LocalMemAllocOp::verify() {
  // MemRefType memrefType = getOriginalMemrefType();
  // if (failed(verifyMemoryOpIndexing(
  //         getOperation(),
  //         (*this)->getAttrOfType<AffineMapAttr>(getMapAttrStr()),
  //         getMapOperands(), memrefType,
  //         /*numIndexOperands=*/getNumOperands() - 1)))
  //   return failure();

  // if (getOriginalMemrefType().getElementType() != getResultType().getElementType())
  //   return emitOpError(
  //       "requires 2 memref types of the same elemental type");

  return success();
}


SmallVector<std::string> LocalMemAllocOp::getKernelNameAsStrVector() {
  Attribute KnNameAttr = this->getOperation()->getAttr("KernelName");
  if(KnNameAttr == nullptr)
    return SmallVector<std::string>();
  else{
    SmallVector<std::string> tokens;
    std::string wholeattr = KnNameAttr.cast<StringAttr>().strref().str();
    size_t start = 0;
    size_t end = 0;
    while ((end = wholeattr.find('+', start)) != std::string::npos) {
        tokens.push_back(wholeattr.substr(start, end - start));
        start = end + 1; 
    }
    std::string lastToken = wholeattr.substr(start);
    if (!lastToken.empty()) { 
      tokens.push_back(lastToken);
    }
    return tokens;
  }
}    
std::string LocalMemAllocOp::addAnotherKernelName(const std::string& newKernel) {
  std::string original = getKernelName().str();
  if (original.empty()) {
    setKernelName(newKernel);
    return newKernel;
  }
  
  setKernelName(original + "+" + newKernel);
  return original + "+" + newKernel;
}

// void AffineLoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
//                                                MLIRContext *context) {
//   results.add<SimplifyAffineOp<AffineLoadOp>>(context);
// }
//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// KernelCallOp
//===----------------------------------------------------------------------===//

LogicalResult KernelCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return LogicalResult::success();

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

FunctionType KernelCallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}


//===----------------------------------------------------------------------===//
// IselOp
//===----------------------------------------------------------------------===//
void IselOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in){
  ::mlir::Type resulttype = in.getType();
  build(odsBuilder, odsState, resulttype, in);
}

LogicalResult IselOp::verify() {
  if(getIn().getType() != getOut().getType())
    return emitOpError(
        "isel op: input should be the same with output.");

  return success();
}


//===----------------------------------------------------------------------===//
// MergeOp
//===----------------------------------------------------------------------===//
void MergeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs){
  assert(inputs.size() > 1);
  ::mlir::Type intype = inputs[0].getType();
  for(auto input : inputs){
    assert(input.getType() == intype && "All inputs of merge op should be the same type.");
  }

  SmallVector<int64_t> shape;
  shape.push_back(inputs.size());

  ::mlir::Type outtype = VectorType::get(shape, intype);
  build(odsBuilder, odsState, outtype, inputs);
}

LogicalResult MergeOp::verify() {
  if(!(getInputs().size() > 1)){
    return emitOpError(
        "merge op: input number of merge op should be larger than 1");
  }
  ::mlir::Type intype = getInput(0).getType();
  for(auto input : getInputs()){
    if(input.getType() != intype){
      return emitOpError(
        "merge op: all inputs of merge op should be the same type.");
    }
  }

  ::mlir::Type outtype = ::llvm::cast<VectorType>(getOut().getType()).getElementType();
  if(outtype != intype){
    return emitOpError(
        "merge op: input and output element type should be the same");
  }

  llvm::ArrayRef<int64_t> shape = ::llvm::cast<VectorType>(getOut().getType()).getShape();
  int num = 1;
  for(auto s : shape){
    num *= s;
  }

  if(num != getInputs().size()){
    return emitOpError(
        "merge op: the count of inputs must correspond to the sum of the element count in the output vectors");
  }

  return success();
}

void MergeOp::print(OpAsmPrinter &p) {
  Type intype = getElementType();
  p << " " ;
  /// print inputs
  llvm::interleaveComma(getInputs(), p, [&](auto it) {
    p << it;
  });

  p << " : " ;
  llvm::interleaveComma(getInputs(), p, [&](auto it) {
    p << intype;
  });
  p << " -> " ;
  p << getOutVectorType();

  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult MergeOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  Type outType;

  /// parse inputs
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs;
  if (parser.parseOperandList(inputs))
    return failure();
  
  /// parse inputs type
  SmallVector<Type, 3> types;
  if (parser.parseColonTypeList(types))
    return failure();

  /// zip inputs and type
  for (auto pair : llvm::zip(inputs, types)){
    if (parser.resolveOperand(std::get<0>(pair),std::get<1>(pair), result.operands))
      return failure();
  }

  /// parse output type
  if (parser.parseArrowTypeList(result.types))
  {
    return failure();
  }

  return parser.parseOptionalAttrDict(result.attributes);
}

#define GET_OP_CLASSES
#include "RAAA/Dialect/ADORA/IR/ADORAOps.cpp.inc"
