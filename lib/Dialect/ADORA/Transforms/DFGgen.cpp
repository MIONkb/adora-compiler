//===----------------------------------------------------------------------===//
//
// This file implements Data flow graph generation
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/OperationSupport.h"
// #include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Support/LLVM.h"

#include <iostream>
#include <string>
#include <bit>
#include "RAAA/Dialect/ADORA/IR/ADORA.h"
#include "PassDetail.h"
#include "RAAA/Dialect/ADORA/Transforms/Passes.h"
#include "RAAA/Dialect/ADORA/Transforms/SimplifyLoadStore.h"
#include "../../../DFG/inc/mlir_cdfg.h"
#include "RAAA/Misc/DFG.h"

// For Block handle 
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

// For op transformation
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

// DFG
ADORA::KernelOp* _kernel_toDFG;
int _variable_config_cnt = 0;

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;

#define DEBUG_TYPE "adora-dfg-gen"

namespace
{
  class ADORALoopCdfgGenPass : public ADORALoopCdfgGenBase<ADORALoopCdfgGenPass>
  {
    void runOnOperation() override;
  };
} // namespace


/// @brief 
/// @param yieldop 
/// @param value 
/// @return the value must be in yieldop's operands
static int GetYieldIndexFromValue(affine::AffineYieldOp yield, mlir::Value v){
  int outidx;
  for(outidx = 0; outidx < yield.getOperands().size(); outidx++){
    if(yield.getOperand(outidx) == v){
      break;
    }
  }
  // assert(outidx < yield.getOperands().size() && "Outer affinefor do not yield this op.");
  if(outidx == yield.getOperands().size())
    return -1;
  else
    return outidx;
}

/// @brief 
/// @param yieldop 
/// @param value 
/// @return the index of the value in yieldop's operands
static int getYieldIndexOfValue(affine::AffineYieldOp yieldop, mlir::Value value){
  int index;
  for(index = 0; index < yieldop.getOperands().size(); index++ ){
    if(value == yieldop.getOperand(index))
      break;
  }
  if(index == yieldop.getOperands().size()){
    LLVM_DEBUG(llvm::errs() << value << "is not an operand of yieldop: " << yieldop);

    assert(isa<affine::AffineForOp>(value.getDefiningOp()->getParentOp()));
    affine::AffineForOp parentforop = dyn_cast<affine::AffineForOp>(value.getDefiningOp()->getParentOp());
    assert(parentforop != dyn_cast<affine::AffineForOp>(yieldop.getOperation()->getParentOp()));

    AffineYieldOp yieldInThisLevel = dyn_cast<AffineYieldOp>(parentforop.getBody()->getTerminator());
    int yieldindex = getYieldIndexOfValue(yieldInThisLevel, value);
    mlir::Value OuterValue = parentforop.getResult(yieldindex);
    return getYieldIndexOfValue(yieldop, OuterValue);
  }
  else{
    return index;
  }
}

/// @brief 
/// @param forop 
/// @param value 
/// @return the index of the value in forop's operands
static int getInitIndexOfValue(affine::AffineForOp forop, mlir::Value value){
  int index;
  for(index = 0; index < forop.getInits().size(); index++ ){
    if(value == forop.getInits()[index])
      break;
  }
  if(index == forop.getInits().size()){
    assert(0);
    // LLVM_DEBUG(llvm::errs() << value << " is not an operand of yieldop: " << forop);

    // assert(isa<affine::AffineForOp>(value.getDefiningOp()->getParentOp()));
    // affine::AffineForOp parentforop = dyn_cast<affine::AffineForOp>(value.getDefiningOp()->getParentOp());
    // assert(parentforop != dyn_cast<affine::AffineForOp>(yieldop.getOperation()->getParentOp()));

    // AffineYieldOp yieldInThisLevel = dyn_cast<AffineYieldOp>(parentforop.getBody()->getTerminator());
    // int yieldindex = getYieldIndexOfValue(yieldInThisLevel, value);
    // mlir::Value OuterValue = parentforop.getResult(yieldindex);
    // return getYieldIndexOfValue(yieldop, OuterValue);
    return -1;
  }
  else{
    return index;
  }
}

/// @brief 
/// @param value 
/// @param op 
/// @return check whether the value is in op's operands
static bool ValueIsInOperands(mlir::Value value, mlir::Operation* op){
  int index;
  for(index = 0; index < op->getOperands().size(); index++ ){
    if(value == op->getOperand(index))
      return true;
  }
  return false;
}

static void SetACCOperandIdx(LLVMCDFGNode* node){
  assert(node->isAcc());
  assert(node->inputNodes().size() <= 2);
  for(auto innode : node->inputNodes()){
    if(isa<AffineForOp>(innode->operation())){
      continue;
    }
    else{
      node->setInputIdx(innode, 0);
    }
  }
}


template <typename DataT>
DataT DataAttrValue2NewType(mlir::Attribute constattr){
  DataT result;
  if(isa<FloatAttr>(constattr)){
    FloatAttr floatattr = dyn_cast<FloatAttr>(constattr);
    double value = floatattr.getValueAsDouble();
    if(floatattr.getType().isF64()){
      result = (DataT)value;
    } 
    else if(floatattr.getType().isF32()){
      result = (DataT)(float) value;
    }
  } 
  else if(isa<IntegerAttr>(constattr))
  {
    IntegerAttr intattr = dyn_cast<IntegerAttr>(constattr);
    if(intattr.getType().isInteger(16)){    
      int value = intattr.getInt();   
      int16_t value_16 = (int16_t) value;       
      result = (DataT)value_16;
    }
    else if(intattr.getType().isInteger(32)){
      int value = intattr.getInt();   
      int32_t value_32 = (int32_t) value;       
      result = (DataT)value_32;
    }
    else if(intattr.getType().isInteger(64)){ 
      int value = intattr.getInt();   
      int64_t value_64 = (int64_t) value;       
      result = (DataT)value_64;
    }
    else if(intattr.getType().isUnsignedInteger(16)){    
      unsigned int value = intattr.getUInt();   
      int16_t value_16 = (int16_t) value;       
      result = (DataT)value_16;
    }
    else if(intattr.getType().isUnsignedInteger(32)){
      unsigned int value = intattr.getUInt();   
      int32_t value_32 = (int32_t) value;       
      result = (DataT)value_32;
    }
    else if(intattr.getType().isUnsignedInteger(64)){ 
      unsigned int value = intattr.getUInt();   
      int64_t value_64 = (int64_t) value;       
      result = (DataT)value_64;
    }
  }
  else if(isa<BoolAttr>(constattr))
  {
    BoolAttr boolattr = dyn_cast<BoolAttr>(constattr);
    bool value = boolattr.getValue();
    result = (DataT)value;
  }
  return result;
}

/////
// A tool function to remove constant truncf
// void RemoveConstantTruncF(func::FuncOp Func){
void RemoveConstantTruncF(ADORA::KernelOp kernel){
  OpBuilder b(kernel);
  kernel.walk([&](arith::TruncFOp truncf){
    Operation* in = truncf.getIn().getDefiningOp();
    mlir::Type outtype = truncf.getOut().getType();
    // in->dump();
    // llvm::errs() << "value: "<< truncf.getOut() << " getType: " << truncf.getOut().getType();

    if(isa<arith::ConstantOp>(in)){
      arith::ConstantOp constin = dyn_cast<arith::ConstantOp>(in);
      mlir::Attribute constattr = in->getAttr(constin.getValueAttrName());
      // llvm::errs() << "constattr: "<< constattr << "\n";
      arith::ConstantOp newconst;
      if(outtype.isF32()){
        float r = DataAttrValue2NewType<float>(constattr);
        FloatAttr outAttr = FloatAttr::get(outtype, r);
        newconst = b.create<arith::ConstantOp>(kernel.getLoc(), outtype , outAttr);
      }
      else if(outtype.isF64()){
        double r = DataAttrValue2NewType<double>(constattr);
        FloatAttr outAttr = FloatAttr::get(outtype, r);
        newconst = b.create<arith::ConstantOp>(kernel.getLoc(), outtype , outAttr);
      }
      else if(outtype.isInteger(16)){
        int16_t r = DataAttrValue2NewType<int16_t>(constattr);
        IntegerAttr outAttr = IntegerAttr::get(outtype, r);
        newconst = b.create<arith::ConstantOp>(kernel.getLoc(), outtype , outAttr);
      }
      else if(outtype.isInteger(32)){
        int32_t r = DataAttrValue2NewType<int32_t>(constattr);
        IntegerAttr outAttr = IntegerAttr::get(outtype, r);
        newconst = b.create<arith::ConstantOp>(kernel.getLoc(), outtype , outAttr);
      }
      else if(outtype.isInteger(64)){
        int64_t r = DataAttrValue2NewType<int64_t>(constattr);
        IntegerAttr outAttr = IntegerAttr::get(outtype, r);
        newconst = b.create<arith::ConstantOp>(kernel.getLoc(), outtype , outAttr);
      }
      // else if(isa<IntegerAttr>(constattr)){

      // }
      // kernel.getOperation()->getBlock()->push_back(newconst);
      newconst.getOperation()->moveAfter(truncf);
      truncf.getOperation()->replaceAllUsesWith(newconst);
    }
  //    
  //     mlir::Attribute constattr = op->getAttr(constop.getValueAttrName());
  //
  //   }
  });
  // kernel.dump();
}

/// @brief A tool function to get another operand's index for a binary operation
/// @param Op 
/// @param operandA 
/// @return index for another operand
static unsigned getAnotherOperandIdx(mlir::Operation* op, mlir::Value operandA){
  assert(op->getOperands().size() == 2);
  int idxA;
  for(idxA = 0; idxA < op->getOperands().size(); idxA++){
    if(op->getOperands()[idxA] == operandA){
      break;
    }
  }
  assert(idxA != op->getOperands().size() && "This operandA value doesn't belong to this op.");
  return op->getOperands().size() - 1 - idxA;
}

// static bool checkAccumulationChain(affine::AffineForOp forop, mlir::Value tocheck, int yieldidx = -1);

/// @brief A tool function to check whether this op is in an accumulation chain.
///     A acc chain is like: blockarg-addf-addf-addf-yield
/// @param forop 
/// @param tocheck 
/// @return true or false
template <typename AccT>
static bool checkAccumulationChain(affine::AffineForOp forop, mlir::Value tocheck, int yieldidx = -1){
  forop.dump();
  SmallVector<mlir::Operation*> uses = getAllUsesInBlock(tocheck, forop.getBody());
  // RegionIterArg.dump();
  if(uses.size() != 1){
    return false;
  }
  else {
    mlir::Operation* use = uses[0];
    use->dump();
    if(isa<AccT>(use))
      return checkAccumulationChain<AccT>(forop, dyn_cast<AccT>(use).getResult(), yieldidx);

    else if(isa<affine::AffineYieldOp>(use)){
      assert(yieldidx!=-1 || dyn_cast<affine::AffineYieldOp>(use).getOperands().size() == 1);
      if(dyn_cast<affine::AffineYieldOp>(use).getOperand(yieldidx) == tocheck)
        return true;
      else
        return false;
    }

    else if(isa<affine::AffineForOp>(use)){
      affine::AffineForOp innerforop = dyn_cast<affine::AffineForOp>(use);
      int index = getInitIndexOfValue(innerforop, tocheck);
      assert(index != -1);

      mlir::Value carry_v = innerforop.getRegionIterArgs()[index];
      carry_v.dump();
      return checkAccumulationChain<AccT>(innerforop, carry_v, index);
    }

    else 
      return false;
  }
}

/// @brief A tool function to check whether this op is in an accumulation chain.
///     A acc chain is like: blockarg-addf-addf-addf-yield
/// @param forop 
/// @param tocheck 
/// @return true or false
template <typename AccT>
static bool checkAccumulationChain(affine::AffineForOp forop, int operandidx){
  mlir::Value blockarg_v = forop.getRegionIterArgs()[operandidx];
  blockarg_v.dump();

  SmallVector<mlir::Operation*> uses = getAllUsesInBlock(blockarg_v, forop.getBody());
  // RegionIterArg.dump();
  if(uses.size() != 1){
    return false;
  }
  else {
    mlir::Operation* use = uses[0];
    if(isa<AccT>(use))
      return checkAccumulationChain<AccT>(forop, dyn_cast<AccT>(use).getResult(), operandidx);
    else if(isa<affine::AffineYieldOp>(use)){
      /// blockarg_v has been directly yield.
      return false;
    }
    // else if(isa<affine::AffineForOp>(use)){
    //   affine::AffineForOp innerforop = dyn_cast<affine::AffineForOp>(use);
    //   int index = getInitIndexOfValue(innerforop, blockarg_v);
    //   assert(index != -1);
    //   return checkAccumulationChain<AccT>(innerforop, index);
    // }
    else 
      return false;
  }
}

/// @brief A tool function to check whether the operation chain between load and store op is in an accumulation chain.
///     A acc chain is like: loadop-addf-addf-addf-storeop
/// @param beused 
/// @param region 
/// @return true or false
template <typename AccT>
static bool checkAccumulationChain(mlir::Operation* op, affine::AffineStoreOp storeop){
  if(op->getNumResults() != 1)
    return false;
  
  AffineForOp forop = dyn_cast<AffineForOp>(op->getParentOp());
  SmallVector<mlir::Operation*> uses = getAllUsesInBlock(op->getResult(0), forop.getBody());
  // RegionIterArg.dump();
  if(uses.size() != 1){
    return false;
  }
  else {
    mlir::Operation* use = uses[0];
    if(isa<AccT>(use))
      return checkAccumulationChain<AccT>(use, storeop);
    else if(isa<affine::AffineYieldOp>(use)){
      //// check whether this yield is used by storeop
      int yieldidx = GetYieldIndexFromValue(dyn_cast<affine::AffineYieldOp>(use), op->getResult(0));
      assert(yieldidx != -1);
      if(storeop.getValue() == forop.getResult(yieldidx))
        return true;
      else
        return false;
    }
    else 
      return false;
  }

  return false;
}

// template <typename AccT>
// static bool checkAccumulationChain(affine::AffineLoadOp loadop, affine::AffineStoreOp storeop){
//   return checkAccumulationChain<AccT>(loadop.getOperation(), storeop);
// }

/// @brief hoist load store op to get loop-carried value, which helps acc extractions 
///   ACC operations: ADD MUL ADDF MULF SEL
/// @param kernel 
void HoistLoadStoreInKernelOp(ADORA::KernelOp kernel){
  HoistLoadStoreOpsInOp(kernel);
  // bool NoChange = 0;
  // while(!NoChange){ // Keep walking the func until no change occurs in this func
  //   NoChange = 1;
  //   SmallVector<AffineLoadOp,  4> ToHoistLoads;
  //   SmallVector<AffineStoreOp, 4> ToHoistStores;
  //   /////////
  //   /// Step 1 : Get all loads op to be hoisted
  //   /////////
  //   kernel.walk([&](AffineLoadOp loadop)
  //   {
  //     Operation* ParentOp = loadop.getOperation()->getParentOp();
  //     if(ParentOp->getName().getStringRef() == AffineForOp::getOperationName() )
  //     { 
  //       AffineForOp ParentForOp = dyn_cast<AffineForOp>(*ParentOp);
  //       // MemRefRegion memrefRegion(Func.getLoc());
  //       // mlir::Value memref;
  //       // SmallVector<mlir::Value, 4> IVs;
  //       for(mlir::Value index : loadop.getIndices()){
  //         if(index == ParentForOp.getInductionVar()){
  //           // This load op can't be hoisted because it is constrained by loop of its level. 
  //           return WalkResult::advance();
  //         }
  //       }
  //       // This load op can be hoisted because it is not constrained by IV of its parent fopOp. 
  //       ToHoistLoads.push_back(loadop);
  //       // llvm::errs() << "[info] move: " << loadop << "\n";
  //     }
  //     return WalkResult::advance();
  //   });

  //   /////////
  //   /// Step 2 : Get all stores op to be hoisted
  //   /////////
  //   kernel.walk([&](AffineStoreOp storeop)
  //   {
  //     // llvm::errs() << "[info] -----------------------------------\n";
  //     // llvm::errs() << "[info] func: \n" << Func << "\n";      
  //     // llvm::errs() << "[info] storeop: " << storeop << "\n";
  //     Operation* ParentOp = storeop.getOperation()->getParentOp();
  //     if(ParentOp->getName().getStringRef() == AffineForOp::getOperationName() )
  //     { 
  //       AffineForOp ParentForOp = dyn_cast<AffineForOp>(*ParentOp);
  //       // MemRefRegion memrefRegion(Func.getLoc());
  //       // mlir::Value memref;
  //       // SmallVector<mlir::Value, 4> IVs;
  //       for(mlir::Value index : storeop.getIndices()){
  //         if(index == ParentForOp.getInductionVar()){
  //           // This store op can't be hoisted because it is constrained by loop of its level. 
  //           return WalkResult::advance();
  //         }
  //       }
  //       // This storeop can be hoisted because it is not constrained by IV of its parent fopOp. 
  //       // construct a loop-carried varaible
  //       ToHoistStores.push_back(storeop);
  //     }
  //     return WalkResult::advance();
  //   });

  //   /////////
  //   /// Step 3 : Do hoists for load-store pairs
  //   /////////
  //   SmallVector<AffineLoadOp,  4> ToHoistLoads_copy = ToHoistLoads;
  //   SmallVector<AffineStoreOp, 4> ToHoistStores_copy = ToHoistStores;
  //   for(AffineLoadOp loadop : ToHoistLoads_copy){
  //     /// Check whether this load occurs with a corresponding store which have
  //     /// the same memref and address to access. 
  //     /// If so, this load-store pair 
  //     /// should be hoisted while construct a loop-carried variable.
  //     for(AffineStoreOp storeop : ToHoistStores_copy){
  //       // llvm::errs() << "[info] loadop: " << loadop << "\n";  
  //       // llvm::errs() << "[info] storeop: " << storeop << "\n"; 
  //       if(ADORA::LoadStoreSameMemAddr(loadop, storeop) 
  //         && checkAccumulationChain<arith::AddIOp>(loadop, storeop)
  //         && checkAccumulationChain<arith::AddFOp>(loadop, storeop)
  //         && checkAccumulationChain<arith::MulIOp>(loadop, storeop)
  //         && checkAccumulationChain<arith::MulFOp>(loadop, storeop)
  //         && checkAccumulationChain<arith::SelectOp>(loadop, storeop)
  //       ){
  //         AffineLoadOp* it_ld;
  //         AffineStoreOp* it_st;   
  //         PositionRelationInLoop PosRelation =
  //               getPositionRelationship(loadop.getOperation(), storeop.getOperation());
  //         switch (PosRelation)
  //         {
  //         case PositionRelationInLoop::SameLevel :
  //         {
  //           // Loadop and store op are in same level so 
  //           // both should be hoisted.
  //           NoChange = 0;

  //           /// Get mlir::Value to be yielded
  //           mlir::Value toYield = storeop.getValue();

  //           /// Get ValueRanges of old for op
  //           SmallVector<mlir::Value, 4> dupIterOperands, dupIterArgs, dupYieldOperands;
  //           Operation* ParentOp = loadop.getOperation()->getParentOp();
  //           if(isa<ADORA::KernelOp>(ParentOp))
  //             continue;

  //           AffineForOp oldForOp = dyn_cast<AffineForOp>(*ParentOp);
  //           OpBuilder builder(oldForOp.getContext());
  //           ValueRange oldIterOperands = oldForOp.getInits();
  //           // ValueRange oldIterArgs = oldForOp.getRegionIterArgs();
  //           ValueRange oldYieldOperands =
  //               cast<AffineYieldOp>(oldForOp.getBody()->getTerminator()).getOperands();
  //           // dupIterOperands.append(oldIterOperands.begin(), oldIterOperands.end());
  //           // dupIterArgs.append(oldIterArgs.begin(), oldIterArgs.end());
  //           dupYieldOperands.append(oldYieldOperands.begin(), oldYieldOperands.end());

  //           // /// Add new mlir::Value to be yielded to dupIterOperands and dupYieldOperands
  //           dupIterOperands.push_back(toYield);
  //           // dupIterArgs.push_back(toYield);
  //           dupYieldOperands.push_back(toYield);

  //           // // Create a new loop with additional iterOperands, iter_args and yield
  //           // // operands. This new loop will take the loop body of the original loop.
  //           // AffineForOp newForOp = replaceForOpWithNewYields(
  //           //     builder, oldForOp, dupIterOperands, dupYieldOperands, dupIterArgs); 
  //           // oldForOp.getOperation()->erase();
  //           IRRewriter rewriter(oldForOp->getContext());

  //           AffineForOp newForOp =
  //             cast<AffineForOp>(*oldForOp.replaceWithAdditionalYields(
  //               rewriter, dupIterOperands, /*replaceInitOperandUsesInLoop=*/false,
  //               [&](OpBuilder &b, Location loc, ArrayRef<BlockArgument> newBbArgs) {
  //                 return dupYieldOperands;
  //               }));
 
  //           // Move load-store pair
  //           loadop.getOperation()->moveBefore(newForOp);           
  //           storeop.getOperation()->moveAfter(newForOp);
      
  //           // Change the input of the new forop
  //           unsigned newOperandIndex = newForOp.getOperation()->getNumOperands() - 1;
  //           newForOp.getOperation()->setOperand(newOperandIndex, loadop);

  //           // Replace all uses of loadop with new iter_arg of forop
  //           replaceAllUsesInRegionWith( loadop.getResult(), 
  //                                       newForOp.getRegionIterArgs()[newOperandIndex],
  //                                       newForOp.getRegion());

  //           // Change the input of the store op
  //           unsigned newResultIndex = newForOp.getOperation()->getNumResults() - 1;
  //           mlir::Value newForResult = newForOp.getOperation()->getResult(newResultIndex);
  //           storeop.getOperation()
  //                     ->setOperand(storeop.getStoredValOperandIndex(),newForResult);
  //           // llvm::errs() << "[info] -----------------------------------\n";
  //           // llvm::errs() << "[info] after move func: \n" << Func << "\n";   
  //           // llvm::errs() << "[info] after newForOp: \n" << newForOp << "\n";     

  //           // Remove hoisted load/store ops from to-check vector
  //           it_ld = std::find(ToHoistLoads.begin(), ToHoistLoads.end(), loadop);
  //           assert(it_ld != ToHoistLoads.end());
  //           ToHoistLoads.erase(it_ld);
  //           it_st = std::find(ToHoistStores.begin(), ToHoistStores.end(), storeop);
  //           assert(it_st != ToHoistStores.end());
  //           ToHoistStores.erase(it_st);
  //           break;
  //         }

  //         case PositionRelationInLoop::LhsOuter :
  //         {
  //           /// load op is in outer level
  //           /// hoist store op only
  //           NoChange = 0;
  //           Operation* storeParentOp = storeop.getOperation()->getParentOp();
  //           if(isa<ADORA::KernelOp>(storeParentOp))
  //             continue;
  //           storeop.getOperation()->moveAfter(storeParentOp);
            
  //           // Remove hoisted load/store ops from to-check vector
  //           it_ld = std::find(ToHoistLoads.begin(), ToHoistLoads.end(), loadop);
  //           assert(it_ld != ToHoistLoads.end());
  //           ToHoistLoads.erase(it_ld);
  //           it_st = std::find(ToHoistStores.begin(), ToHoistStores.end(), storeop);
  //           assert(it_st != ToHoistStores.end());
  //           ToHoistStores.erase(it_st);
  //           break;
  //         }

  //         case PositionRelationInLoop::RhsOuter : 
  //         {
  //           /// store op is in outer level
  //           /// hoist load op only
  //           NoChange = 0;
  //           Operation* loadopParentOp = loadop.getOperation()->getParentOp();
  //           loadop.getOperation()->moveBefore(loadopParentOp);
  //           if(isa<ADORA::KernelOp>(loadopParentOp))
  //             continue;

  //           // Remove hoisted load/store ops from to-check vector
  //           it_ld = std::find(ToHoistLoads.begin(), ToHoistLoads.end(), loadop);
  //           assert(it_ld != ToHoistLoads.end());
  //           ToHoistLoads.erase(it_ld);
  //           it_st = std::find(ToHoistStores.begin(), ToHoistStores.end(), storeop);
  //           assert(it_st != ToHoistStores.end());
  //           ToHoistStores.erase(it_st);
  //           break;
  //         }
          
  //         default:
  //           break;
  //         }
  //       }
  //     }
  //   }

  //   /////////
  //   /// Step 4 : Do hoists for remaining load/store 
  //   /////////
  //   for(AffineLoadOp loadop : ToHoistLoads){
  //     /// If the loadop has no corresponding store then just hoist.
  //     Operation* ParentOp = loadop.getOperation()->getParentOp();
  //     if(isa<ADORA::KernelOp>(ParentOp))
  //       continue;

  //     assert(ParentOp->getName().getStringRef() == AffineForOp::getOperationName());
  //     NoChange = 0;
  //     loadop.getOperation()->moveBefore(ParentOp);
  //   }    
  //   for(AffineStoreOp storeop : ToHoistStores){
  //     Operation* ParentOp = storeop.getOperation()->getParentOp();
  //     if(isa<ADORA::KernelOp>(ParentOp))
  //       continue;

  //     assert(ParentOp->getName().getStringRef() == AffineForOp::getOperationName());
  //     NoChange = 0;
  //     storeop.getOperation()->moveAfter(ParentOp);
  //   }
  // }  
}

/**
 * A tool function to check whether a mlir value is the initial value of a accumulation chain.
 * */
template <typename AccT>
bool IsAccumulationInitialValue(affine::AffineForOp forop, mlir::Value InitValue){
  int idx;
  for(idx = 0; idx < forop.getInits().size(); idx++){
    if(forop.getInits()[idx] == InitValue){
      break;
    }
  }
  assert(idx != forop.getInits().size() && "This init value doesn't belong to this for op.");
  mlir::Value RegionIterArg = forop.getRegionIterArgs()[idx];
  SmallVector<mlir::Operation*> uses = getAllUsesInBlock(RegionIterArg, forop.getBody());
  // RegionIterArg.dump();
  if(uses.size() != 1){
    /// not an accumulation chain.
    /// TODO: really?
    return false;
  }
  else{
    mlir::Operation* use = uses[0];
    if(isa<affine::AffineForOp>(use)){
      /// an initial value of loop carried value of inner affine for.
      affine::AffineForOp innerforop = dyn_cast<affine::AffineForOp>(use);
      return IsAccumulationInitialValue<AccT>(innerforop, RegionIterArg);
    }
    else if(isa<AccT>(use)){
      return checkAccumulationChain<AccT>(forop, RegionIterArg, idx);
    }
    else {
      return false;
    }
  }
}

/**
 * A tool function to move the initial value computing of accumulation to the outer most level.
 * */
template <typename AccT>
bool MoveAccumulationInitialValue(affine::AffineForOp forop, mlir::Value InitValue){
  int idx;
  for(idx = 0; idx < forop.getInits().size(); idx++){
    if(forop.getInits()[idx] == InitValue){
      break;
    }
  }
  assert(idx != forop.getInits().size() && "This init value doesn't belong to this for op.");
  OpBuilder b(forop);
  mlir::Type datatype = InitValue.getType();
  arith::ConstantOp newconst;
  if(datatype.isF32()){
    float constvalue = 0;
    FloatAttr constAttr = FloatAttr::get(datatype, constvalue);
    newconst = b.create<arith::ConstantOp>(forop.getLoc(), datatype , constAttr);
  }
  else if(datatype.isF64()){
    double constvalue = 0;
    FloatAttr constAttr = FloatAttr::get(datatype, constvalue);
    newconst = b.create<arith::ConstantOp>(forop.getLoc(), datatype , constAttr);
  }
  else {
    int64_t constvalue = 0;
    IntegerAttr constAttr = IntegerAttr::get(datatype, constvalue);
    newconst = b.create<arith::ConstantOp>(forop.getLoc(), datatype , constAttr);    
  }
  // datatype.dump();
  // forop.dump();
  // forop.getOperation()->getBlock()->dump();
  InitValue.getDefiningOp()->replaceAllUsesWith(newconst);
  AccT newcompute = b.create<AccT>(forop.getLoc(), forop.getResults()[idx] , InitValue); 
  newcompute.getOperation()->moveAfter(forop);
  // forop.dump();
  // forop.getOperation()->getBlock()->dump();

  forop.getResults()[idx].replaceAllUsesWith(newcompute);
  newcompute.setOperand(0, forop.getResults()[idx]);

  // forop.getOperation()->getBlock()->dump();
  // forop.dump();
}

/**
 * 
 * A tool function to move the initial value computing of accumulation to the outer most level.
 * For example:
              affine.for %arg7 = 0 to 28 {
                %4 = affine.load %2[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
                %5 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %4) -> (f32) {
                  %6 = affine.for %arg10 = 0 to 7 iter_args(%arg11 = %arg9) -> (f32) {
                    %7 = affine.load %0[0, %arg8, %arg10, %arg7 * 2] : memref<1x3x7x62xf32>
                    %8 = affine.load %1[0, %arg8, %arg10, 0] : memref<2x3x7x7xf32>
                    %9 = arith.mulf %7, %8 : f32
                    %10 = arith.addf %arg11, %9 : f32
                    affine.yield %34 : f32
                  }
                  affine.yield %6 : f32
                }
                affine.store %5, %3[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
              }
 *  Can change to:
 *            affine.for %arg7 = 0 to 28 {
                %4 = affine.load %2[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
                %cst = arith.const 0 : f32
                %5 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %cst) -> (f32) {
                  %6 = affine.for %arg10 = 0 to 7 iter_args(%arg11 = %arg9) -> (f32) {
                    %7 = affine.load %0[0, %arg8, %arg10, %arg7 * 2] : memref<1x3x7x62xf32>
                    %8 = affine.load %1[0, %arg8, %arg10, 0] : memref<2x3x7x7xf32>
                    %9 = arith.mulf %7, %8 : f32
                    %10 = arith.addf %arg11, %9 : f32
                    affine.yield %34 : f32
                  }
                  affine.yield %6 : f32
                }
                %6 = %10 = arith.addf %cst, %5 : f32
                affine.store %6, %3[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
              }
 * 
*/
void MoveLoopCarriedInitailValue(ADORA::KernelOp kernel){
  // OpBuilder b(kernel);
  kernel.walk([&](affine::AffineForOp forop){
    for(mlir::Value IterOperand : forop.getInits()){
      // IterOperand.dump();
      AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
      if(!IterOperand.isa<BlockArgument>() 
        && yieldop.getOperands().size() != 0){
        if(IsAccumulationInitialValue<arith::AddIOp>(forop, IterOperand) 
          && !IterOperand.getDefiningOp<arith::ConstantOp>()){
          MoveAccumulationInitialValue<arith::AddIOp>(forop, IterOperand);
        }
        else if(IsAccumulationInitialValue<arith::AddFOp>(forop, IterOperand)
          && !IterOperand.getDefiningOp<arith::ConstantOp>()){
          MoveAccumulationInitialValue<arith::AddFOp>(forop, IterOperand);
        }
        // else if(IsAccumulationInitialValue<arith::MulIOp>(forop, IterOperand)){
        //   MoveAccumulationInitialValue<arith::MulIOp>(forop, IterOperand);;
        // }
        // else if(IsAccumulationInitialValue<arith::MulFOp>(forop, IterOperand)){
        //   MoveAccumulationInitialValue<arith::MulFOp>(forop, IterOperand);;
        // }
      }
    }
  });
  // kernel.dump();
}



/**
 * 
 * A tool function to For accumulation chain, move accumulation operation to the last using commutative law of addition/multiplication
 * For example:
              affine.for %arg7 = 0 to 28 {
                %4 = affine.load %2[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
                %5 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %4) -> (f32) {
                  %6 = affine.for %arg10 = 0 to 7 iter_args(%arg11 = %arg9) -> (f32) {
                    %7 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c0] : memref<1x3x7x62xf32>
                    %8 = affine.load %1[0, %arg8, %arg10, %c0] : memref<2x3x7x7xf32>
                    %9 = arith.mulf %7, %8 : f32
                    %10 = arith.addf %arg11, %9 : f32
                    %c1 = arith.constant 1 : index
                    %11 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c1] : memref<1x3x7x62xf32>
                    %12 = affine.load %1[0, %arg8, %arg10, %c1] : memref<2x3x7x7xf32>
                    %13 = arith.mulf %11, %12 : f32
                    %14 = arith.addf %10, %13 : f32
                    %c2 = arith.constant 2 : index
                    %15 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c2] : memref<1x3x7x62xf32>
                    %16 = affine.load %1[0, %arg8, %arg10, %c2] : memref<2x3x7x7xf32>
                    %17 = arith.mulf %15, %16 : f32
                    %18 = arith.addf %14, %17 : f32
                    %c3 = arith.constant 3 : index
                    %19 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c3] : memref<1x3x7x62xf32>
                    %20 = affine.load %1[0, %arg8, %arg10, %c3] : memref<2x3x7x7xf32>
                    %21 = arith.mulf %19, %20 : f32
                    %22 = arith.addf %18, %21 : f32
                    %c4 = arith.constant 4 : index
                    %23 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c4] : memref<1x3x7x62xf32>
                    %24 = affine.load %1[0, %arg8, %arg10, %c4] : memref<2x3x7x7xf32>
                    %25 = arith.mulf %23, %24 : f32
                    %26 = arith.addf %22, %25 : f32
                    %c5 = arith.constant 5 : index
                    %27 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c5] : memref<1x3x7x62xf32>
                    %28 = affine.load %1[0, %arg8, %arg10, %c5] : memref<2x3x7x7xf32>
                    %29 = arith.mulf %27, %28 : f32
                    %30 = arith.addf %26, %29 : f32
                    %c6 = arith.constant 6 : index
                    %31 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c6] : memref<1x3x7x62xf32>
                    %32 = affine.load %1[0, %arg8, %arg10, %c6] : memref<2x3x7x7xf32>
                    %33 = arith.mulf %31, %32 : f32
                    %34 = arith.addf %30, %33 : f32
                    affine.yield %34 : f32
                  }
                  affine.yield %6 : f32
                }
                affine.store %5, %3[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
              }
 *  Can change to:
              affine.for %arg7 = 0 to 28 {
                %4 = affine.load %2[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
                %5 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %4) -> (f32) {
                  %6 = affine.for %arg10 = 0 to 7 iter_args(%arg11 = %arg9) -> (f32) {
                    %7 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c0] : memref<1x3x7x62xf32>
                    %8 = affine.load %1[0, %arg8, %arg10, %c0] : memref<2x3x7x7xf32>
                    %9 = arith.mulf %7, %8 : f32

                    %c1 = arith.constant 1 : index
                    %11 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c1] : memref<1x3x7x62xf32>
                    %12 = affine.load %1[0, %arg8, %arg10, %c1] : memref<2x3x7x7xf32>
                    %13 = arith.mulf %11, %12 : f32
                    %14 = arith.addf %9, %13 : f32
                    %c2 = arith.constant 2 : index
                    %15 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c2] : memref<1x3x7x62xf32>
                    %16 = affine.load %1[0, %arg8, %arg10, %c2] : memref<2x3x7x7xf32>
                    %17 = arith.mulf %15, %16 : f32
                    %18 = arith.addf %14, %17 : f32
                    %c3 = arith.constant 3 : index
                    %19 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c3] : memref<1x3x7x62xf32>
                    %20 = affine.load %1[0, %arg8, %arg10, %c3] : memref<2x3x7x7xf32>
                    %21 = arith.mulf %19, %20 : f32
                    %22 = arith.addf %18, %21 : f32
                    %c4 = arith.constant 4 : index
                    %23 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c4] : memref<1x3x7x62xf32>
                    %24 = affine.load %1[0, %arg8, %arg10, %c4] : memref<2x3x7x7xf32>
                    %25 = arith.mulf %23, %24 : f32
                    %26 = arith.addf %22, %25 : f32
                    %c5 = arith.constant 5 : index
                    %27 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c5] : memref<1x3x7x62xf32>
                    %28 = affine.load %1[0, %arg8, %arg10, %c5] : memref<2x3x7x7xf32>
                    %29 = arith.mulf %27, %28 : f32
                    %30 = arith.addf %26, %29 : f32
                    %c6 = arith.constant 6 : index
                    %31 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c6] : memref<1x3x7x62xf32>
                    %32 = affine.load %1[0, %arg8, %arg10, %c6] : memref<2x3x7x7xf32>
                    %33 = arith.mulf %31, %32 : f32
                    %34 = arith.addf %30, %33 : f32

                    %10 = arith.addf %arg11, %34 : f32
                    affine.yield %10 : f32
                  }
                  affine.yield %6 : f32
                }
                affine.store %5, %3[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
              }
 * 
*/
void MoveAccumulationToLast(ADORA::KernelOp kernel){
  OpBuilder b(kernel);
  kernel.walk([&](affine::AffineForOp forop){
    int IterRegionOperandIdx;
    for(IterRegionOperandIdx = 0; IterRegionOperandIdx < forop.getNumRegionIterArgs(); IterRegionOperandIdx++){
      mlir::Value IterRegionOperand = forop.getRegionIterArgs()[IterRegionOperandIdx];
      if(checkAccumulationChain<arith::AddFOp>(forop, IterRegionOperand, IterRegionOperandIdx)){
        /// Adjust consumer of iter operand
        assert(getAllUsesInBlock(IterRegionOperand, forop.getBody()).size() == 1);
        mlir::Operation* IterArgConsumer = getAllUsesInBlock(IterRegionOperand, forop.getBody())[0];
        // IterArgConsumer->dump();
        // if(isa<affine::AffineForOp>(IterArgConsumer->getParentOp()))
        //   continue;
        assert(isa<arith::AddFOp>(IterArgConsumer));
        mlir::Value AnotherOperand = IterArgConsumer->getOperand(getAnotherOperandIdx(IterArgConsumer, IterRegionOperand));
        IterArgConsumer->replaceAllUsesWith(AnotherOperand.getDefiningOp());
        IterArgConsumer->erase();

        /// Adjust producer of yield
        AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
        mlir::Value OldYieldProducer_Value = yieldop.getOperand(IterRegionOperandIdx);
        mlir::Operation* OldYieldProducer = OldYieldProducer_Value.getDefiningOp();
        arith::AddFOp NewYieldProducer = b.create<arith::AddFOp>(OldYieldProducer->getLoc(), OldYieldProducer_Value, IterRegionOperand); 
        NewYieldProducer.getOperation()->moveBefore(yieldop);
        OldYieldProducer->replaceAllUsesWith(NewYieldProducer);
        NewYieldProducer.setOperand(0, OldYieldProducer_Value);
      }
      else if(checkAccumulationChain<arith::AddIOp>(forop, IterRegionOperand, IterRegionOperandIdx)){
        /// Adjust consumer of iter operand
        assert(getAllUsesInBlock(IterRegionOperand, forop.getBody()).size() == 1);
        mlir::Operation* IterArgConsumer = getAllUsesInBlock(IterRegionOperand, forop.getBody())[0];

        if(isa<affine::AffineForOp>(IterArgConsumer->getParentOp()))
          continue;   
        
        assert(isa<arith::AddIOp>(IterArgConsumer));
        mlir::Value AnotherOperand = IterArgConsumer->getOperand(getAnotherOperandIdx(IterArgConsumer, IterRegionOperand));
        IterArgConsumer->replaceAllUsesWith(AnotherOperand.getDefiningOp());
        IterArgConsumer->erase();

        /// Adjust producer of yield
        AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
        mlir::Value OldYieldProducer_Value = yieldop.getOperand(IterRegionOperandIdx);
        mlir::Operation* OldYieldProducer = OldYieldProducer_Value.getDefiningOp();
        arith::AddIOp NewYieldProducer = b.create<arith::AddIOp>(OldYieldProducer->getLoc(), OldYieldProducer_Value, IterRegionOperand); 
        NewYieldProducer.getOperation()->moveBefore(yieldop);
        OldYieldProducer->replaceAllUsesWith(NewYieldProducer);
        NewYieldProducer.setOperand(0, OldYieldProducer_Value);
      }
      else if(checkAccumulationChain<arith::MulFOp>(forop, IterRegionOperand, IterRegionOperandIdx)){
        /// Adjust consumer of iter operand
        assert(getAllUsesInBlock(IterRegionOperand, forop.getBody()).size() == 1);
        mlir::Operation* IterArgConsumer = getAllUsesInBlock(IterRegionOperand, forop.getBody())[0];
        
        if(isa<affine::AffineForOp>(IterArgConsumer->getParentOp()))
          continue;  

        assert(isa<arith::MulFOp>(IterArgConsumer));
        mlir::Value AnotherOperand = IterArgConsumer->getOperand(getAnotherOperandIdx(IterArgConsumer, IterRegionOperand));
        IterArgConsumer->replaceAllUsesWith(AnotherOperand.getDefiningOp());
        IterArgConsumer->erase();

        /// Adjust producer of yield
        AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
        mlir::Value OldYieldProducer_Value = yieldop.getOperand(IterRegionOperandIdx);
        mlir::Operation* OldYieldProducer = OldYieldProducer_Value.getDefiningOp();
        arith::MulFOp NewYieldProducer = b.create<arith::MulFOp>(OldYieldProducer->getLoc(), OldYieldProducer_Value, IterRegionOperand); 
        NewYieldProducer.getOperation()->moveBefore(yieldop);
        OldYieldProducer->replaceAllUsesWith(NewYieldProducer);
        NewYieldProducer.setOperand(0, OldYieldProducer_Value);
      }
      else if(checkAccumulationChain<arith::MulIOp>(forop, IterRegionOperand, IterRegionOperandIdx)){
        /// Adjust consumer of iter operand
        assert(getAllUsesInBlock(IterRegionOperand, forop.getBody()).size() == 1);
        mlir::Operation* IterArgConsumer = getAllUsesInBlock(IterRegionOperand, forop.getBody())[0];
        
        if(isa<affine::AffineForOp>(IterArgConsumer->getParentOp()))
          continue;  
                  
        assert(isa<arith::MulIOp>(IterArgConsumer));
        mlir::Value AnotherOperand = IterArgConsumer->getOperand(getAnotherOperandIdx(IterArgConsumer, IterRegionOperand));
        IterArgConsumer->replaceAllUsesWith(AnotherOperand.getDefiningOp());
        IterArgConsumer->erase();

        /// Adjust producer of yield
        AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
        mlir::Value OldYieldProducer_Value = yieldop.getOperand(IterRegionOperandIdx);
        mlir::Operation* OldYieldProducer = OldYieldProducer_Value.getDefiningOp();
        arith::MulIOp NewYieldProducer = b.create<arith::MulIOp>(OldYieldProducer->getLoc(), OldYieldProducer_Value, IterRegionOperand); 
        NewYieldProducer.getOperation()->moveBefore(yieldop);
        OldYieldProducer->replaceAllUsesWith(NewYieldProducer);
        NewYieldProducer.setOperand(0, OldYieldProducer_Value);
      }
      // else if(checkAccumulationChain<arith::MulIOp>(forop, IterRegionOperand, IterRegionOperandIdx)){
      //   /// Adjust consumer of iter operand
      //   mlir::Operation* IterArgConsumer = getAllUsesInBlock(IterRegionOperand, forop.getBody())[0];
      //   assert(isa<arith::MulIOp>(IterArgConsumer));
      //   mlir::Value AnotherOperand = IterArgConsumer->getOperand(getAnotherOperandIdx(IterArgConsumer, IterRegionOperand));
      //   IterArgConsumer->replaceAllUsesWith(AnotherOperand.getDefiningOp());
      //   IterArgConsumer->erase();

      //   /// Adjust producer of yield
      //   AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
      //   mlir::Value OldYieldProducer_Value = yieldop.getOperand(IterRegionOperandIdx);
      //   mlir::Operation* OldYieldProducer = OldYieldProducer_Value.getDefiningOp();
      //   arith::MulIOp NewYieldProducer = b.create<arith::MulIOp>(OldYieldProducer->getLoc(), OldYieldProducer_Value, IterRegionOperand); 
      //   NewYieldProducer.getOperation()->moveBefore(yieldop);
      //   OldYieldProducer->replaceAllUsesWith(NewYieldProducer);
      //   NewYieldProducer.setOperand(0, OldYieldProducer_Value);
      // }
      else{
        //// Insert ISEL operation
        // (void)ReplaceLoopCarryValueWithNewIselOp(forop, IterRegionOperandIdx);
        continue;
      }
      // else if(checkAccumulationChain<arith::SelectOp>(forop, IterRegionOperand)){
      //   /// Adjust consumer of iter operand
      //   mlir::Operation* IterArgConsumer = getAllUsesInBlock(IterRegionOperand, forop.getBody())[0];
      //   assert(isa<arith::SelectOp>(IterArgConsumer));
      //   mlir::Value AnotherOperand = IterArgConsumer->getOperand(getAnotherOperandIdx(IterArgConsumer, IterRegionOperand));
      //   IterArgConsumer->replaceAllUsesWith(AnotherOperand.getDefiningOp());
      //   IterArgConsumer->erase();

      //   /// Adjust producer of yield
      //   AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
      //   mlir::Value OldYieldProducer_Value = yieldop.getOperand(IterRegionOperandIdx);
      //   mlir::Operation* OldYieldProducer = OldYieldProducer_Value.getDefiningOp();
      //   arith::SelectOp NewYieldProducer = b.create<arith::SelectOp>(OldYieldProducer->getLoc(), OldYieldProducer_Value, IterRegionOperand); 
      //   NewYieldProducer.getOperation()->moveBefore(yieldop);
      //   OldYieldProducer->replaceAllUsesWith(NewYieldProducer);
      //   NewYieldProducer.setOperand(0, OldYieldProducer_Value);
      // }
    }
    //   // IterOperand.dump();
    //   AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
    //   if(!IterOperand.isa<BlockArgument>() 
    //     && yieldop.getOperands().size() != 0){
    //     if(IsAccumulationInitialValue<arith::AddIOp>(forop, IterOperand)){
    //       MoveAccumulationInitialValue<arith::AddIOp>(forop, IterOperand);
    //     }
    //     else if(IsAccumulationInitialValue<arith::AddFOp>(forop, IterOperand)){
    //       MoveAccumulationInitialValue<arith::AddFOp>(forop, IterOperand);;
    //     }
    //   }
    // }
  });
  // kernel.dump();
}


/**
 * 
 * A tool function to Insert ISEL operator if init-xxxxx-yield chain is not complete
 * For example:
              xxx
 * 
 * There are two circumstances that need to add Isel to yield op.
 * 1: If RegionOperand is yield immediately, and IterRegionOperandIdx is not paired with yield
 * 2: If compute op before yield is select
*/
void InsertIselForLoopCarry(ADORA::KernelOp kernel, bool verbose){
  OpBuilder b(kernel);
  kernel.walk([&](affine::AffineForOp forop){
    if(verbose) forop.dump();
    int IterRegionOperandIdx;
    for(IterRegionOperandIdx = 0; IterRegionOperandIdx < forop.getNumRegionIterArgs(); IterRegionOperandIdx++){
      mlir::Value IterRegionOperand = forop.getRegionIterArgs()[IterRegionOperandIdx];
      if(verbose) IterRegionOperand.dump();

      /// get yieldop 
      AffineYieldOp yieldop = dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());

      /// check RegionOperand is yield immediately
      int YieldIndex = GetYieldIndexFromValue(yieldop, IterRegionOperand);

      /// If IterRegionOperandIdx and yield is not paired
      if(YieldIndex != -1 && IterRegionOperandIdx != YieldIndex){
        (void)ReplaceLoopCarryValueWithNewIselOp(forop, IterRegionOperandIdx);
      }
      /// If compute op before yield is not acc type, insert isel op
      else if(YieldIndex == -1
          && !checkAccumulationChain<arith::AddIOp>(forop, IterRegionOperand, IterRegionOperandIdx)
          && !checkAccumulationChain<arith::AddFOp>(forop, IterRegionOperand, IterRegionOperandIdx)
          && !checkAccumulationChain<arith::MulIOp>(forop, IterRegionOperand, IterRegionOperandIdx)
          && !checkAccumulationChain<arith::MulFOp>(forop, IterRegionOperand, IterRegionOperandIdx)){
        (void)ReplaceLoopCarryValueWithNewIselOp(forop, IterRegionOperandIdx);
      }
    }
  });
  // kernel.dump();
}

SmallVector<affine::AffineForOp> SortForVec_InToOutLevels(SmallVector<affine::AffineForOp> ForVec){
  SmallVector<affine::AffineForOp> NewForVec;
  // bubble sort
  for(int i = 0; i < ForVec.size() - 1; i ++){
    for(int j = 0; j < ForVec.size() - 1 - i; j++){ 
      //  ForVec[i].dump();
      //  ForVec[j].dump();
      auto result = ForVec[j].walk([&](affine::AffineForOp f)-> WalkResult 
      {
        if(f == ForVec[j + 1])// loop j + 1 is inside loop j, put j + 1 to tail
          return WalkResult::interrupt();
        else
          return WalkResult::advance();
      });
      if(result == WalkResult::interrupt()){
        // change position
        auto temp = ForVec[j];
        ForVec[j] = ForVec[j + 1];
        ForVec[j + 1] = temp;
      }
    } 
  }
  // for(auto for_ : ForVec){
  //   for_.dump();
  // }
  return ForVec;
}

std::string LinearAccessToStr(mlir::SmallVector<std::pair<int64_t, int64_t>> Vec){
  std::string str;
  for(auto elem : Vec){
    str += std::to_string(elem.first) + "," +  std::to_string(elem.second) + ",";
    // llvm::errs() << str;
  }
  return str.substr(0, str.length() - 1); // delete the last ", "
}

std::string LinearAccessToStr(mlir::SmallVector<std::pair<std::string, std::string>> Vec){
  std::string str;
  for(auto elem : Vec){
    str += elem.first + "," +  elem.second + ",";
    // llvm::errs() << str;
  }
  return str.substr(0, str.length() - 1); // delete the last ", "
}


template <typename LoadOrStoreOp>
  int64_t GetInitAddr(LoadOrStoreOp lsop, std::map<mlir::Operation*, int> For_loop_level){
  OpBuilder b(lsop);
  Operation::operand_range loadIndices = lsop.getIndices();
  ::mlir::AffineMap map = lsop.getAffineMapAttr().getValue();

  MemRefType memRefType = lsop.getMemref().getType().template cast<MemRefType>();
  ArrayRef<int64_t>  Shape = memRefType.getShape();
  int64_t ElementBytes = memRefType.getElementTypeBitWidth()/8;
  // map.dump();
  SmallDenseMap<AffineForOp, int64_t> ForToLb;
  mlir::SmallVector<int64_t> initPosition_eachrank;
  for(unsigned r = 0; r < map.getResults().size(); r++ ){
    AffineExpr expr = map.getResult(r);
    AffineMap lb_new_map;
    SmallVector<AffineExpr> dim_to_expr;
    for(unsigned d = 0; d < loadIndices.size(); d++){
      // loadIndices[d].dump();
      if(expr.isFunctionOfDim(d)){
        if (loadIndices[d].isa<BlockArgument>()){
          //// a block arguement of for op
          AffineForOp forop = dyn_cast<AffineForOp>(loadIndices[d].getParentBlock()->getParentOp());
          assert(isa<AffineForOp>(forop) && "AffineLoadOp or StoreOp 's parent op should be AffineForOp!");
          // get lower bound of this dim
          assert(forop.getLowerBoundMap().getResults().size() == 1);
          AffineExpr lbExpr = forop.getLowerBoundMap().getResult(0);
          assert(lbExpr.getKind() == AffineExprKind::Constant);
          int64_t lb = lbExpr.dyn_cast<AffineConstantExpr>().getValue();
          dim_to_expr.push_back(b.getAffineConstantExpr(lb));
        }
        else if(!loadIndices[d].isa<BlockArgument>() && isa<arith::ConstantOp>(loadIndices[d].getDefiningOp())){
          arith::ConstantOp constop = dyn_cast<arith::ConstantOp>(loadIndices[d].getDefiningOp());
          mlir::Attribute constattr = loadIndices[d].getDefiningOp()->getAttr(constop.getValueAttrName());
          assert(isa<IntegerAttr>(constattr));
          IntegerAttr intattr = dyn_cast<IntegerAttr>(constattr);
          int64_t value = intattr.getInt();   
          dim_to_expr.push_back(b.getAffineConstantExpr(value));        
        }
        else if(!loadIndices[d].isa<BlockArgument>() && isa<affine::AffineApplyOp>(loadIndices[d].getDefiningOp())){
          affine::AffineApplyOp applyop = dyn_cast<affine::AffineApplyOp>(loadIndices[d].getDefiningOp());
          mlir::AffineMap applymap = applyop.getAffineMap();
          // loadIndices[d].dump();
          // applymap.dump();
          assert(applymap.getResults().size() == 1 && applymap.getNumDims() == 1);
          // mlir::AffineExpr constpart = getConstPartofAffineExpr(applymap.results()[0]);
          SmallVector<AffineExpr, 4> dimReplacements(applymap.getNumDims());
          for(unsigned d = 0; d < applyop.getMapOperands().size(); d++){
            mlir::Value operand = applyop.getMapOperands()[d];
            assert(operand.isa<BlockArgument>());
            AffineForOp forop = dyn_cast<AffineForOp>(operand.getParentBlock()->getParentOp());
            assert(isa<AffineForOp>(forop) && "AffineLoadOp or StoreOp 's parent op should be AffineForOp!");
            // get lower bound of this dim
            assert(forop.getLowerBoundMap().getResults().size() == 1);
            AffineExpr lbExpr = forop.getLowerBoundMap().getResult(0);
            assert(lbExpr.getKind() == AffineExprKind::Constant);
            // int64_t lb = lbExpr.dyn_cast<AffineConstantExpr>().getValue();
            dimReplacements[d] = lbExpr;
          }
          applymap = applymap.replaceDimsAndSymbols(dimReplacements, {}, applymap.getNumDims(), applymap.getNumSymbols());
          // applymap.dump();
          dim_to_expr.push_back(applymap.getResults()[0]);
        }
        else{
          assert(0 && "Only supported arith::ConstantOp and affineForOp now.");
        }

      }
      else {
        dim_to_expr.push_back(b.getAffineDimExpr(d));
      }
    }
    // lb_new_map = 
    lb_new_map = map.compose(AffineMap::get(loadIndices.size(), 0, ArrayRef<AffineExpr>(dim_to_expr), b.getContext()));
    // llvm::errs() << "[test] lb_new_map: " ; lb_new_map.dump();
    assert(lb_new_map.getResult(r).getKind() == AffineExprKind::Constant);
    int64_t init_position_thisrank = lb_new_map.getResult(r).dyn_cast<AffineConstantExpr>().getValue();
    initPosition_eachrank.push_back(init_position_thisrank);
  }

  int64_t init_position = 0;
  for(unsigned r = 0; r < Shape.size(); r++ ){
    int64_t elements_each_step_inner = 1;
    for (unsigned i = r + 1; i < Shape.size(); i++){
      // llvm::errs()<<"Shape[i]: " << Shape[i] << "\n";
      elements_each_step_inner *= Shape[i];
    }
    init_position += initPosition_eachrank[r] * elements_each_step_inner;
    // llvm::errs()<<"init_position: " << init_position 
    //             <<",initPosition[r]: " << initPosition_eachrank[r] 
    //             <<",elements_each_step_inner: " << elements_each_step_inner
    //             << "\n";
  }
  // return LinearAccess;
  return init_position * ElementBytes;
}

template <typename LoadOrStoreOp>
  int64_t GetMemrefSize(LoadOrStoreOp lsop){
  MemRefType memRefType = lsop.getMemref().getType().template cast<MemRefType>();
  ArrayRef<int64_t>  Shape = memRefType.getShape();
  int64_t ElementBytes = memRefType.getElementTypeBitWidth()/8;
  // map.dump();

  int64_t elements = 1;
  for(unsigned r = 0; r < Shape.size(); r++ ){
    elements *= Shape[r];
  }

  return elements * ElementBytes;
}

/// Get the string of compare operation's type, for example:
///      %7 = arith.cmpf ugt, %arg10, %6 : f32
/// return: ugt
std::string GetCMPTypeStr(mlir::Operation* op){
  std::string cmptype;
  if(isa<arith::CmpIOp>(op)){
    arith::CmpIOp cmpop = dyn_cast <arith::CmpIOp> (op);
    arith::CmpIPredicate cmppred = cmpop.getPredicate();
    cmptype = stringifyCmpIPredicate(cmppred);
    if(cmptype == "eq") return "EQ";
    else if(cmptype == "ne") return "NE";
    else if(cmptype == "ult") return "ULT";
    else if(cmptype == "ule") return "ULE";
    else if(cmptype == "ugt") return "UGT";
    else if(cmptype == "uge") return "UGE";
    else assert(0 && "Unsupported compare type.");
  }
  else if(isa<arith::CmpFOp>(op)){
    arith::CmpFOp cmpop = dyn_cast <arith::CmpFOp> (op);
    arith::CmpFPredicate cmppred = cmpop.getPredicate();
    cmptype = stringifyCmpFPredicate(cmppred);
    if(cmptype == "ueq") return "FEQ32";
    else if(cmptype == "une") return "FNE32";
    else if(cmptype == "ugt") /*return "FUGT32";*/return "FOGT32";
    else if(cmptype == "uge") /*return "FUGE32";*/return "FOGE32";
    else if(cmptype == "ult") /*return "FULT32";*/return "FOLT32";
    else if(cmptype == "ule") /*return "FULE32";*/return "FOLE32";
    /*CGRA only support ordered float computing in current version.*/

    else if(cmptype == "oeq") return "FEQ32";
    else if(cmptype == "one") return "FNE32";
    else if(cmptype == "ogt") return "FOGT32";
    else if(cmptype == "oge") return "FOGE32";
    else if(cmptype == "olt") return "FOLT32";
    else if(cmptype == "ole") return "FOLE32";

    else if(cmptype == "uno") /*return "FULE32";*/return "FUNO32";

    else assert(0 && "Unsupported compare type.");   
  }
  else 
    assert(0 && "Not a compare operation.");
  
  return cmptype;
}
/// Sometimes the PEs only support "less than(lt)" and "less equal(le)"
/// so we convert greater to less
bool ConvertGreaterToLess(LLVMCDFGNode* node){
  if(node->getTypeName() == "UGT") node->setTypeName("ULT");
  else if(node->getTypeName() == "UGE") node->setTypeName("ULE");
  else if(node->getTypeName() == "FUGT32") node->setTypeName("FULT32");
  else if(node->getTypeName() == "FUGE32") node->setTypeName("FULE32");
  else if(node->getTypeName() == "FOGT32") node->setTypeName("FOLT32");
  else if(node->getTypeName() == "FOGE32") node->setTypeName("FOLE32");
  else return true;

  /// exchange operand idx
  if(node->inputEdges().size() != 2){
    return false;
  }

  auto input0 = node->getInputPort(0);
  auto input1 = node->getInputPort(1);
  node->setInputIdx(input0, 1);
  node->setInputIdx(input1, 0);

  return true;
}

AffineYieldOp getOldestAncestorYieldOp(AffineForOp forop){
  AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
  assert(yieldop.getOperands().size() == 1);
  mlir::Operation* operandop = yieldop.getOperand(0).getDefiningOp();
  if(isa<AffineForOp>(operandop)){
    return getOldestAncestorYieldOp(dyn_cast<AffineForOp>(operandop));
  }
  else {
    return yieldop;
  }
  
}

bool isInteger(const std::string& str) {
    // Empty string or just a negative/positive sign is not a valid integer
    if (str.empty() || ((str[0] == '-' || str[0] == '+') && str.size() == 1)) {
        return false;
    }

    // Check each character to see if it's a digit (allowing an optional leading sign)
    for (size_t i = (str[0] == '-' || str[0] == '+') ? 1 : 0; i < str.size(); ++i) {
        if (!std::isdigit(str[i])) {
            return false;
        }
    }
    return true;
}

#define Define_Polymorphism_Of_StringIntArith(funcname) \
  std::string funcname (const int& LHS, const int& RHS){  \
    return funcname(std::to_string(LHS), std::to_string(RHS)); } \
  std::string funcname (const std::string& LHS, const int& RHS){  \
    return funcname(LHS, std::to_string(RHS)); } \
  std::string funcname (const int& LHS, const std::string& RHS){  \
    return funcname(std::to_string(LHS), RHS); }

////
/// Multiply two parameters from affine for op
///   three situations:
///   1 int * int = int
///   2 arg * int = arg
///   3 arg * arg = arg
std::string MulAsStr (const std::string& LHS, const std::string& RHS){
  if(isInteger(LHS) && isInteger(RHS)){
    return std::to_string(std::stoi(LHS) * std::stoi(RHS));
  }
  else{
    if(std::stoi(LHS) == 1)
      return RHS;
    else if(std::stoi(RHS) == 1)
      return LHS;
    else
      return LHS + "*" + RHS;
  }
}
Define_Polymorphism_Of_StringIntArith(MulAsStr)

////
/// Add two parameters from affine for op
///   three situations:
///   1 int + int = int
///   2 arg + int = arg
///   3 arg + arg = arg
std::string AddAsStr (const std::string& LHS, const std::string& RHS){
  if(isInteger(LHS) && isInteger(RHS)){
    return std::to_string(std::stoi(LHS) + std::stoi(RHS));
  }
  else{
    if(std::stoi(LHS) == 0)
      return RHS;
    else if(std::stoi(RHS) == 0)
      return LHS;
    else
      return LHS + "+" + RHS;
  }
}
Define_Polymorphism_Of_StringIntArith(AddAsStr)

////
/// Add two parameters from affine for op
///   four situations:
///   1 int - int = int
///   2 arg - int = arg
///   3 int - arg = arg
///   4 arg - arg = arg
std::string SubAsStr (const std::string& LHS, const std::string& RHS){
  if(isInteger(LHS) && isInteger(RHS)){
    return std::to_string(std::stoi(LHS) - std::stoi(RHS));
  }
  else{
    if(std::stoi(LHS) == 0)
      return "-" + RHS;
    else if(std::stoi(RHS) == 0)
      return LHS;
    else
      return LHS + "-" + RHS;
  }
}
Define_Polymorphism_Of_StringIntArith(SubAsStr)




//// Get the outter level of one operation in one kernel
std::string getOuterLoopTotalTripcountUntilKernel(mlir::Operation* op){
  // std::string total_tripcount_str = "1";
  int tripcount = 1;
  if(isa<affine::AffineForOp>(op->getParentOp())){
    tripcount = getConstantTripCount(dyn_cast<affine::AffineForOp>(op->getParentOp())).value_or(0);
    assert(tripcount != 0);

    // if(isInteger(total_tripcount_str)) 
    //   total_tripcount = std::stoi(total_tripcount_str);

    return MulAsStr(std::to_string(tripcount), getOuterLoopTotalTripcountUntilKernel(op->getParentOp()));
  }
  else if(isa<ADORA::KernelOp>(op->getParentOp())){
    return std::to_string(1);
  }
  else{
    assert(false && "ADORA Kernel is wrong.");
  }
}

//// Get the trip count of forop, return as a string, for example:
///     affine.for %arg7 = 0 to 28 return 28
///     affine.for %arg1 = 0 to %arg2 return "arg2"
///     for nonconstant for op, we can only handle following 2 types:
///       affine.for %arg1 = 0 to %arg2
///       affine.for %arg1 = 0 to 2000 - %arg2
static std::string getTripCountAsStr(affine::AffineForOp forop){
  OpBuilder b(forop);
  int tripcount = getConstantTripCount(forop).value_or(0); 
  if(tripcount == 0){
    /// Non-const iteration space
    AffineMap map;
    SmallVector<mlir::Value> operands;
    getTripCountMapAndOperands(forop, &map, &operands);
    LLVM_DEBUG(
      llvm::errs() << "[MION] Non-const iteration space: ";
      map.dump();
      for(auto operand : operands){
        llvm::errs() << operand << " ";
      }
      llvm::errs() << "\n";
    );
    assert(map.getNumDims() + map.getNumSymbols() == 1 && "Unsupported trip count!");
    assert(map.getNumResults() == 1 && operands.size() == 1 && "Unsupported trip count!");
    AffineExpr expr = map.getResult(0);
    LLVM_DEBUG(expr.dump(););
    switch (expr.getKind())
      {
      case AffineExprKind::DimId : 
      case AffineExprKind::SymbolId : {
        mlir::Value arg_v = operands[0]; ///operands.size() == 1
        // assert(!IsInKernel(arg_v.getOperation()));
        
        Location loc = _kernel_toDFG->getLoc(); ////// Fix this
        // loc.dump();
        // LLVM_DEBUG(_kernel_toDFG->dump(););
        b.setInsertionPointToStart(_kernel_toDFG->getOperation()->getBlock());
        arith::ConstantOp cst = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(b.getIndexType(), 0));
        arith::AddIOp newadd = b.create<arith::AddIOp>(loc, arg_v, cst);
        LLVM_DEBUG(newadd.getOperation()->getBlock()->dump(););

        newadd.getOperation()->moveBefore(_kernel_toDFG->getOperation());
        LLVM_DEBUG(newadd.getOperation()->getBlock()->dump(););
        
        cst.getOperation()->moveBefore(newadd.getOperation());
        LLVM_DEBUG(newadd.getOperation()->getBlock()->dump(););

        //// denote the name of the generated operation
        StringAttr strattr = StringAttr::get(newadd.getOperation()->getContext(),
                                              "VARCFG_" + std::to_string(_variable_config_cnt));
        newadd.getOperation()->setAttr("VAR_CONFIG", strattr);
        // _variable_config_cnt++;
        LLVM_DEBUG(forop.dump(););
        
        return "VARCFG_" + std::to_string(_variable_config_cnt++);
      }

      // case AffineExprKind::Add : ///TODO: FIX THIS

      default :{
        assert(false && "Unsupported trip count!");
        return "";
      }
    }
  }
  else{
    return std::to_string(tripcount);
  }
}


////
// get linear access from load or store op
template <typename LoadOrStoreOp>
  mlir::SmallVector<std::pair<std::string, std::string>> \
    GetLinearAccess(LoadOrStoreOp lsop, std::map<mlir::Operation*, int> For_loop_level){
  Operation::operand_range loadIndices = lsop.getIndices();
  ::mlir::AffineMap map = lsop.getAffineMapAttr().getValue();
  MemRefType memRefType = lsop.getMemref().getType().template cast<MemRefType>();
  // map.dump();
  SmallDenseMap<AffineForOp, SmallVector<int64_t>> ForToRanks;
  // SmallVector<AffineForOp> forVec;
  for(unsigned d = 0; d < loadIndices.size(); d++){
    if(!loadIndices[d].isa<BlockArgument>() && isa<arith::ConstantOp>(loadIndices[d].getDefiningOp())){
      /// Constant index bias doesn't contribute to linear access.
      continue;
    }
    // llvm::errs() << "[test] loadIndice[i]: " ; loadIndices[d].dump() ; 
    AffineForOp forop = dyn_cast<AffineForOp>(loadIndices[d].getParentBlock()->getParentOp());
    assert(isa<AffineForOp>(forop) && "AffineLoadOp or StoreOp 's parent op should be AffineForOp!");
    // AffineForOp parentFor = dyn_cast<AffineForOp>(*forop);
    // llvm::errs() << "[test] forop: " ; forop.dump();

    /// For every dim of the affine map,  add the corresponding Multiplicator to ForToRanks
    for(unsigned r = 0; r < map.getResults().size(); r++ ){
      AffineExpr expr = map.getResult(r);
      // expr.dump();
      if(expr.isFunctionOfDim(d)){
        // find the corresponding rank
        ForToRanks[forop].push_back(ADORA::MultiplicatorOfDim(expr, d));
        // llvm::errs() << d <<": " << ADORA::MultiplicatorOfDim(expr, d) << ",";
      } 
      else {
        ForToRanks[forop].push_back(0);
      }
    }
    // assert(findElement(forVec, forop)==-1 && "For op should only be in the indices for one time.");
    // forVec.push_back(forop);
  }
  // forVec = SortForVec_InToOutLevels(forVec);

  mlir::SmallVector<std::pair<std::string, std::string>> LinearAccess;
  for(unsigned level = 0; level < For_loop_level.size(); level ++){
    /// For a new recursion of this level,
    ///   elements_each_step = RM * STEP * RANK_SHAPE
    affine::AffineForOp forop;
    for(auto loop_level : For_loop_level){
      if(loop_level.second == level){
        assert(isa<affine::AffineForOp>(loop_level.first) && "We can only handle affine for now.");
        forop = dyn_cast<affine::AffineForOp>(loop_level.first);
        break;
      }
    }
    SmallVector<int64_t> RankMultiplicators = ForToRanks[forop];
    int64_t ElementBytes = memRefType.getElementTypeBitWidth()/8;
    // std::string tripcount = getConstantTripCount(forop).value_or(0); 
    std::string tripcount = getTripCountAsStr(forop);
    // LLVM_DEBUG(_kernel_toDFG->dump(););
    // total_count_str = MulAsStr(total_count_str, tripcount);
    // LLVM_DEBUG(_kernel_toDFG->dump(););

    int64_t elements_each_step = 0;
    bool RM_flag = false;
    for(unsigned r = 0; r < RankMultiplicators.size(); r++){
      if(RankMultiplicators[r] == 0){
        continue;
      }
      else{
        assert(RM_flag == false && "This for loop should only be corresponding to one rank.");
        elements_each_step = forop.getStep() * RankMultiplicators[r];
        ArrayRef<int64_t>  Shape = memRefType.getShape();
        for (unsigned i = r + 1; i < Shape.size(); i++){
          elements_each_step *= Shape[i];
        }
        RM_flag = true;
      }
    }

    /// For the last old recursion ,
    //   end_position = lb0 + (tripcount0-1) * step0 * rm0 * rank0 + lb1 + (tripcount1-1) * step1 * rm1 * rank1 + ...
    std::string end_position = "0";
    for(unsigned innerlevel = 0; innerlevel < level; innerlevel++){
      affine::AffineForOp innerforop;
      for(auto loop_level : For_loop_level){
        if(loop_level.second == innerlevel){
          assert(isa<affine::AffineForOp>(loop_level.first) && "We can only handle affine for now.");
          innerforop = dyn_cast<affine::AffineForOp>(loop_level.first);
          break;
        }
      }
      // int64_t innertripcount = getConstantTripCount(innerforop).value_or(0); 
      std::string innertripcount = getTripCountAsStr(innerforop);
      SmallVector<int64_t> innerRMs = ForToRanks[innerforop];
      
      // get lower bound of this dim
      assert(innerforop.getLowerBoundMap().getResults().size() == 1);
      AffineExpr lbExpr = innerforop.getLowerBoundMap().getResult(0);
      assert(lbExpr.getKind() == AffineExprKind::Constant);
      int64_t lb = lbExpr.dyn_cast<AffineConstantExpr>().getValue();
      // llvm::errs()<< "lbmap: " << lbmap << "\n"; 

      bool innerRM_flag = false;
      int64_t elements_each_step_inner = 0;
      for(unsigned r = 0; r < innerRMs.size(); r++){
        if(innerRMs[r] == 0){
          continue;
        }
        else{
          assert(innerRM_flag == false && "This for loop should only be corresponding to one rank.");
          elements_each_step_inner = innerforop.getStep() * innerRMs[r];
          ArrayRef<int64_t>  Shape = memRefType.getShape();
          for (unsigned i = r + 1; i < Shape.size(); i++){
            // llvm::errs()<<"Shape[i]: " << Shape[i] << "\n";
            elements_each_step_inner *= Shape[i];
          }
          innerRM_flag = true;
        }
      }
      //// end_position += lb + elements_each_step_inner * innertripcount;
      end_position = AddAsStr(end_position, 
                      AddAsStr(lb, 
                        MulAsStr(elements_each_step_inner, 
                          SubAsStr(innertripcount, 1)))); 
      // llvm::errs()<<"end_position: " << end_position 
      //           << ", elements_each_step_inner:" << elements_each_step_inner
      //           << ", lb:" << lb
      //           <<"\n";
    }

    /// addr for new recursion
    std::string addrstep = MulAsStr(SubAsStr(elements_each_step, end_position), ElementBytes);
    LLVM_DEBUG(llvm::errs()<<"addrstep: " << addrstep
                << ", ElementBytes:" << ElementBytes
                << ", elements_each_step:" << elements_each_step
                << ", end_position:" << end_position
                <<"\n";);

    LinearAccess.push_back(std::pair(addrstep, tripcount));
  }
  
  LLVM_DEBUG(llvm::errs()<<"\n" << LinearAccessToStr(LinearAccess)<<"\n";);
  return LinearAccess;
}



////
// Get accumulation information from a yield-for-yield-for.... chain in a recursive method.
// vector count_interval_repeat:
//  $interval: accumate every $interval cycles
//  $count: PE be set to initial value every $count times of accumulation
//  $repeat: above operation will be repeat for $repeat times
SmallVector<std::string, 3> GetACCInfoFromYieldNode(LLVMCDFGNode* yieldnode, SmallVector<std::string, 3>& count_interval_repeat/*, int YieldIndex = 0*/){
  LLVM_DEBUG(_kernel_toDFG->dump(););
  
  assert(count_interval_repeat.size() == 3);
  assert(yieldnode->getTypeName() == "yield");
  // assert(yieldnode->inputNodes().size() == 1);
  assert(yieldnode->outputNodes().size() == 1);

  std::string total_count_str = count_interval_repeat[0];
  std::string total_interval_str = count_interval_repeat[1];
  std::string total_repeat_str = count_interval_repeat[2];
  std::string tripcount_str;

  //// get count
  LLVMCDFGNode* SuccNode = yieldnode->outputNodes()[0];
  assert(SuccNode->getTypeName() == "for");
  AffineForOp forop = dyn_cast<AffineForOp>(SuccNode->operation());
  assert(IsIterationSpaceSupported(forop));
  std::string tripcount = getTripCountAsStr(forop);
  LLVM_DEBUG(_kernel_toDFG->dump(););
  total_count_str = MulAsStr(total_count_str, tripcount);
  LLVM_DEBUG(_kernel_toDFG->dump(););

  // LLVMCDFGNode* AnceNode = yieldnode->inputNodes()[0]; 
  // if(AnceNode->getTypeName() == "for"){
  //   forop = dyn_cast<AffineForOp>(AnceNode->operation());
  //   int tripcount = getConstantTripCount(forop).value_or(0); 
  // }
  // else{
  //   /// This loop level limits the count of acc.
  //   int tripcount = getConstantTripCount(forop).value_or(0); 
  //   if(tripcount == 0){
  //     /// Non-const iteration space

  //   }
  // }

  count_interval_repeat[0] = total_count_str;
  count_interval_repeat[1] = total_interval_str;

  // assert(SuccNode->outputNodes().size() == 1);
  LLVMCDFGNode* NextYieldNode;
  for(LLVMCDFGNode* nextnode : SuccNode->outputNodes()){
    if(!nextnode->isInputBackEdge(SuccNode)){
      /// backedge is a loop-carried variable
      NextYieldNode = nextnode;
      break;
    }
  }
  if(NextYieldNode->getTypeName() == "yield"){
    /// Get the yield index of this yielded value in the outer for level.
    // AffineYieldOp outeryieldop =  dyn_cast<AffineYieldOp>(outerFor.getBody()->getTerminator());
    // int OuterIndex = GetYieldIndexFromValue(dyn_cast<affine::YieldOp>(NextYieldNode->operation()), forop.getResults()[index]);
    return GetACCInfoFromYieldNode(NextYieldNode, count_interval_repeat/*, OuterIndex*/);
  }
  else{
    /// get repeat
    std::string temp = getOuterLoopTotalTripcountUntilKernel(forop.getOperation());
    total_repeat_str = MulAsStr(total_repeat_str, temp);
    count_interval_repeat[2] = total_repeat_str; 
    /// recursion should stop.
    return count_interval_repeat;
  }
}

/// Delete the yield-for-yield-for... chain in CDFG
bool DeleteYield(LLVMCDFG* CDFG, LLVMCDFGNode* yieldnode){
  assert(yieldnode->getTypeName() == "yield");
  assert(yieldnode->outputNodes().size() == 1);
  LLVMCDFGNode* SuccNode = yieldnode->outputNodes()[0];
  assert(SuccNode->getTypeName() == "for");

  for(LLVMCDFGNode* AnceNode : yieldnode->inputNodes()){
    std::map<LLVMCDFGNode*, unsigned> OutputNodeToIdx;
    mlir::Operation* AnceOp = AnceNode->operation();
    
    assert(AnceOp->getResults().size() == 1);
    int YieldIndex = getYieldIndexOfValue(dyn_cast<affine::AffineYieldOp>(yieldnode->operation()), AnceOp->getResult(0));
    
    /// get output nodes
    affine::AffineForOp forop = dyn_cast<affine::AffineForOp>(SuccNode->operation());
    for(LLVMCDFGNode* outputnode : SuccNode->outputNodes()){
      mlir::Operation* OutputOp = outputnode->operation();
      if( !outputnode->isInputBackEdge(SuccNode) && ValueIsInOperands(forop.getResult(YieldIndex), OutputOp)){
        /// backedge is a loop-carried variable
        // AfterForNode = nextnode;
        OutputNodeToIdx[outputnode] = outputnode->getInputIdx(SuccNode);
      }
    }
    
    /// connect AnceNode and output node
    for(auto _pair: OutputNodeToIdx){
      LLVMCDFGNode* outputnode = _pair.first;
      unsigned idx = _pair.second;
      outputnode->addInputNode(AnceNode,  idx, /*isBackEdge=*/false);
      AnceNode->addOutputNode(outputnode, /*isBackEdge=*/false);
      CDFG->addEdge(AnceNode, outputnode);   
    }
  }



  // AfterForNode->operation()->dump();
  // AffineForOp forop = dyn_cast<AffineForOp>(SuccNode->operation());
  // if(AfterForNode->getTypeName() == "yield"){
  // LLVMCDFGNode* NextForNode = AfterForNode->outputNodes()[0];
  // assert(NextForNode->getTypeName() == "for");
  // NextForNode->operation()->dump();

  CDFG->delNode(yieldnode);
  CDFG->delNode(SuccNode);



  
  return true;
  // }
  // else{
  //   return false;
  // }
}


///////
// A data bit_cast function. The data will be organized as a vector<unsigned char> which stores the hex byte.
// String is not suitable because string stores signed char I think.
//////
template <typename DataT>
  std::vector<unsigned char> DataBitCastToHex(DataT data){
  std::stringstream result;
    // std::string a;
  unsigned char *hex = (unsigned char*) (&data);
  unsigned length = sizeof(DataT);
  std::vector<unsigned char> c;
  // llvm::errs() << "\ndata:" << data  << "\n";
  for(int i = 0; i < length; i++){
    c.push_back(uint32_t(hex[i]));
  }
  // std::string str;
  // int str_len = str.size();
  // for(int i = length - 1; i >= 0 ; i--){
  //   result<< std::hex << std::uppercase << std::setfill('0') << std::setw(2) << uint32_t(c[i]);
  //   std::cout << std::hex << std::setw(2)  << uint32_t(c[i]) << std::endl;
  // }
  // llvm::errs() << "\nstr:" << result.str()  << "\n";

  // DataT* data_back = (DataT*)(unsigned char*) &(c[0]);
  // llvm::errs() << "\ndata_back:" << *data_back  << "\n";
  return c;
}




///////
// Get Initial mlir Value from a for op
//////
static mlir::Value getInitialValueFromFor(affine::AffineForOp forop, int index = 0){
  // assert(forop.getNumRegionIterArgs() == 1);
  if(forop.getNumRegionIterArgs() == 1){
    assert(index == 0);
  }

  if(forop.getInits()[index].isa<BlockArgument>()){
    affine::AffineForOp outerFor = dyn_cast<affine::AffineForOp>(forop.getInits()[index].getParentBlock()->getParentOp());
    assert(outerFor && "Outer operation is not affine for.");

    /// Get the yield index of this yielded value in the outer for level.
    AffineYieldOp outeryieldop =  dyn_cast<AffineYieldOp>(outerFor.getBody()->getTerminator());
    int OuterIndex = GetYieldIndexFromValue(outeryieldop, forop.getResults()[index]);
    assert(OuterIndex != -1);

    return getInitialValueFromFor(dyn_cast<affine::AffineForOp>(forop.getOperation()->getParentOp()), OuterIndex);
  }
  else{
    return forop.getInits()[index];
  }
}
static mlir::Value getInitialValueFromYieldIndex(affine::AffineYieldOp yield, int index){
  return getInitialValueFromFor(dyn_cast<affine::AffineForOp>(yield.getOperation()->getParentOp()), index);
}


void setConstantNode(LLVMCDFGNode* node){
  mlir::Operation* op = node->operation();
  arith::ConstantOp constop = dyn_cast<arith::ConstantOp>(op);
  std::string value_str;
  // llvm::errs() << constop.getValue();
  // mlir::Type constTy = constop.getValue().getType();
  mlir::Attribute constattr = op->getAttr(constop.getValueAttrName());
  if(isa<FloatAttr>(constattr)){
    FloatAttr floatattr = dyn_cast<FloatAttr>(constattr);
    double value = floatattr.getValueAsDouble();
    if(floatattr.getType().isF64()){
      // unsigned char *hex = (unsigned char*) (&value);
      // int hex_int = (int)*hex;
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value);
      // llvm::errs() << "value:"<< value << ", hex:"  << ConstCal_hex ;
      node->setConstValHex(ConstCal_hex);
      // llvm::errs() << "value:"<< value << ", hex:" << *hex << ", int:" << hex_int;
      node->setDataBits(64);
    } 
    else if(floatattr.getType().isF32()){
      float value_float = (float) value;
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value_float);
      node->setConstValHex(ConstCal_hex);
      node->setDataBits(32);
    }
  } 
  else if(isa<IntegerAttr>(constattr))
  {
    IntegerAttr intattr = dyn_cast<IntegerAttr>(constattr);
    if(intattr.getType().isInteger(16)){    
      int value = intattr.getInt();   
      int16_t value_16 = (int16_t) value;       
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value_16);
      node->setConstValHex(ConstCal_hex);
      node->setDataBits(16);
    }
    else if(intattr.getType().isInteger(32)){
      int value = intattr.getInt();   
      int32_t value_32 = (int32_t) value;     
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value_32);
      node->setConstValHex(ConstCal_hex);
      node->setDataBits(32);
    }
    else if(intattr.getType().isInteger(64)){ 
      int value = intattr.getInt();           
      int64_t value_64 = (int64_t) value;        
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value_64);
      node->setConstValHex(ConstCal_hex);
      node->setDataBits(64);
    }
    else if(intattr.getType().isUnsignedInteger(16)){    
      unsigned int value = intattr.getUInt();          
      int16_t value_16 = (int16_t) value;       
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value_16);
      node->setConstValHex(ConstCal_hex);
      node->setDataBits(16);
    }
    else if(intattr.getType().isUnsignedInteger(32)){
      unsigned int value = intattr.getUInt();   
      int32_t value_32 = (int32_t) value;     
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value_32);
      node->setConstValHex(ConstCal_hex);
      node->setDataBits(32);
    }
    else if(intattr.getType().isUnsignedInteger(64)){ 
      unsigned int value = intattr.getUInt();                
      int64_t value_64 = (int64_t) value;        
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value_64);
      node->setConstValHex(ConstCal_hex);
      node->setDataBits(64);
    }
  }
  else if(isa<BoolAttr>(constattr))
  {
    BoolAttr boolattr = dyn_cast<BoolAttr>(constattr);
    bool value = boolattr.getValue();
    std::vector<unsigned char> ConstCal_hex;
    ConstCal_hex.push_back((char)value);
    node->setConstValHex(ConstCal_hex);
    node->setDataBits(1);
  }
}

uint32_t ConstantOpToHex(arith::ConstantOp constop){
  mlir::Attribute constattr = constop.getValueAttr();
  unsigned char *hex;
  if(isa<FloatAttr>(constattr)){
    FloatAttr floatattr = dyn_cast<FloatAttr>(constattr);
    double value = floatattr.getValueAsDouble();
    float value_float = (float)value;
    hex = (unsigned char*) (&value_float);
    return (uint32_t)*hex;
  } 
  else if(isa<IntegerAttr>(constattr))
  {
    IntegerAttr intattr = dyn_cast<IntegerAttr>(constattr);
    int value = intattr.getInt(); 
    uint32_t value_32 = (uint32_t) value;
    hex = (unsigned char*) (&value_32);  
    return (uint32_t)*hex;
  }
  else if(isa<BoolAttr>(constattr))
  {
    BoolAttr boolattr = dyn_cast<BoolAttr>(constattr);
    bool value = boolattr.getValue();
    uint32_t value_32 = (uint32_t) value;
    hex = (unsigned char*) (&value_32);  
    return (uint32_t)*hex; 
  }
}

//////////////////////////////
// Handle self-cycles in CDFG. Extract Acc operators and ISEL operators.
//////////////////////////////
static void HandleSelfCycle(LLVMCDFG* CDFG, bool verbose = true){
  auto nodes = CDFG->nodes();
  SmallVector<LLVMCDFGNode*> YieldsToBeDelete;
  for(auto &elem : nodes){
    // int node_id = elem.first;
    LLVMCDFGNode* node = elem.second;
    if(node->getTypeName() == "yield"){
      /// Get the operand of yield op.
      AffineYieldOp yieldop = dyn_cast<AffineYieldOp>(node->operation());
      if(yieldop.getOperands().size() != 0){
        // yieldop.dump();
        mlir::Operation* forop = node->operation()->getParentOp();
        // forop->dump();

        LLVMCDFGNode* fornode = CDFG->node(forop);
        // fornode->addOutputNode(node, /*isBackEdge=*/false);
        // node->addInputNode(fornode,  /*operand_idx=*/ 1, /*isBackEdge=*/false);

        fornode->addInputNode(node,  /*edgeidx=*/0, /*isBackEdge=*/true);
        node->addOutputNode(fornode, /*isBackEdge=*/true);
        CDFG->addEdge(node, fornode); //To fix: Edge Type
        // for(auto &elem : CDFG->edges()){
        //   auto edge = elem.second;
        //   auto srcName = edge->src()->getName();
        //   auto dstName = edge->dst()->getName();
        //   llvm::errs() << srcName << " -> " << dstName << "\n";
        // }
        // CDFG->delNode();
        YieldsToBeDelete.push_back(node);
      } 
      else{
        CDFG->delNode(node);
      }
    }
  }
  if(verbose) { CDFG->CDFGtoDOT(CDFG->name_str()+"_1_CDFG.dot");}
  
  nodes = CDFG->nodes();
  for(auto &elem : nodes){
    // int node_id = elem.first;
    LLVMCDFGNode* node = elem.second;
    if(node->getTypeName() == "yield"){
      /// Get the operand of yield op.
      AffineYieldOp yieldop = dyn_cast<AffineYieldOp>(node->operation());
      AffineForOp forop = dyn_cast<AffineForOp>(yieldop.getOperation()->getParentOp());
      assert(yieldop.getOperands().size() == forop.getOperands().size());

      for(int OperandIdx = 0; OperandIdx < yieldop.getOperands().size(); OperandIdx++){
        LLVMCDFGNode* ComputeNode;
        // mlir::Value init_mlir_value;
        std::string accTypeName;
        uint32_t init_value;
        // if(isa<BlockArgument>(yieldop.getOperand(OperandIdx))){
        //   /// Create a ISEL node
        //   init_mlir_value = getInitialValueFromYieldIndex(yieldop, OperandIdx); 
        //   // ComputeNode = CDFG->addNode("ISEL");
        //   accTypeName = "ISEL";
        // }
        // else{
          mlir::Operation* ComputeOp = yieldop.getOperand(OperandIdx).getDefiningOp();
          if(verbose) llvm::errs() << "[debug]ComputeOp: ";
          if(verbose) ComputeOp->dump();
          ComputeNode = CDFG->node(ComputeOp);
          mlir::Value init_mlir_value = getInitialValueFromYieldIndex(yieldop, OperandIdx);   
          if(verbose) llvm::errs() << "[debug]init_mlir_value: " << init_mlir_value << "\n";
          
          /// set acc type
          if(ComputeNode->getTypeName() == "ADD" 
            && checkAccumulationChain<arith::AddIOp>(forop, OperandIdx)){
            /// ACC
            accTypeName = "ACC";
          }
          else if(ComputeNode->getTypeName() == "FADD32"
            && checkAccumulationChain<arith::AddFOp>(forop, OperandIdx)){
            /// FACC32
            accTypeName = "FACC32";
          }
          else if(ComputeNode->getTypeName() == "MUL"
            && checkAccumulationChain<arith::MulIOp>(forop, OperandIdx)){
            /// MACC
            accTypeName = "MACC";
          }
          else if(ComputeNode->getTypeName() == "FMUL32"
            && checkAccumulationChain<arith::MulFOp>(forop, OperandIdx)){
           /// FMACC32
            accTypeName = "FMACC32";
          }
          else if(ComputeNode->getTypeName() == "SEL"
            && checkAccumulationChain<arith::SelectOp>(forop, OperandIdx)){
            /// SEL can be extracted as accumulation mode as well.
            accTypeName = "ISEL";
          }
          else if(!checkAccumulationChain<ADORA::IselOp>(forop, OperandIdx)
          // && ComputeNode->getTypeName() == "ISEL"
          ){
            accTypeName = "ISEL";
            //// connect init node ------> ISEL <- - - - - - loop carry node 
            ///////// Get ISEL node
            SmallVector<mlir::Operation*> uses = getAllUsesInBlock(forop.getRegionIterArgs()[OperandIdx], forop.getBody());
            assert(uses.size()==1);
            if(!isa<ADORA::IselOp>(uses[0])){
              continue;
            }
            // assert(isa<ADORA::IselOp>(uses[0]));
            ADORA::IselOp iselop = dyn_cast<ADORA::IselOp>(uses[0]);
            auto IselNode = CDFG->node(iselop.getOperation());

            ///////// Connect init node 
            // auto InitNode = CDFG->node(init_mlir_value.getDefiningOp());
            // bool isBackEdge = false;
            // InitNode->addOutputNode(IselNode, isBackEdge);
            // IselNode->addInputNode(InitNode, /*edgeidx=*/1, isBackEdge);
            // CDFG->addEdge(InitNode, IselNode); //To fix: Edge Type

            ///////// connect loop carry node 
            bool isBackEdge = true;
            ComputeNode->addOutputNode(IselNode, isBackEdge);
            IselNode->addInputNode(ComputeNode, /*edgeidx=*/0, isBackEdge);
            LLVMCDFGEdge* backedge = CDFG->addEdge(ComputeNode, IselNode); //To fix: Edge Type
            backedge->setIterDist(1);

            // DependInfo DI;
            // DI.type = 
            // continue;
            ComputeNode = IselNode;
          }
          else{
            continue;
          }
        // }

        //// init value
        if(isa<arith::ConstantOp>(init_mlir_value.getDefiningOp())){
          arith::ConstantOp constop = dyn_cast<arith::ConstantOp>(init_mlir_value.getDefiningOp());
          init_value = ConstantOpToHex(constop);
        }
        else if(isa<affine::AffineLoadOp>(init_mlir_value.getDefiningOp())){
          init_value = 0x00000000;
        } else{
          /// TODO: What to do if it is not constant op
          assert(0);
        }

        SmallVector<std::string, 3> count_interval_repeat = {"1", "1", "1"};///count/interval/repeat
        count_interval_repeat = GetACCInfoFromYieldNode(node, count_interval_repeat);
        ComputeNode->setTypeName(accTypeName);
        ComputeNode->setAcc();
        ComputeNode->setACCinit(std::to_string(init_value));
        ComputeNode->setACCcount(count_interval_repeat[0]);
        ComputeNode->setACCinterval(count_interval_repeat[1]);
        ComputeNode->setACCrepeat(count_interval_repeat[2]);    

        //// Change operand idx of acc op
        SetACCOperandIdx(ComputeNode);
      }
    }
  }
  
  /// Thirdly, delete yield-for nodes
  // unsigned k = 55;
  for(auto ynode : YieldsToBeDelete){
    DeleteYield(CDFG, ynode);
  }
}

static bool HandlCompareNode(LLVMCDFG* CDFG, bool verbose = true){
  auto nodes = CDFG->nodes();
  for(auto &elem : nodes){
    // int node_id = elem.first;
    LLVMCDFGNode* node = elem.second;
    mlir::Operation* op = node->operation();
    // if(verbose) {op->dump();}
    if(   op->getName().getStringRef() == "arith.cmpi"
        ||op->getName().getStringRef() == "arith.cmpf"){
      node->setTypeName(GetCMPTypeStr(op));
      if(!ConvertGreaterToLess(node)){
        return false;
      }
    }
  }
  return true;
}


bool generateCDFGfromKernelAfterOptimization(LLVMCDFG* CDFG, ADORA::KernelOp kernel, bool verbose){
  if(verbose) {kernel.dump();}
  _kernel_toDFG = &kernel;

  int level = 0, level_total;
  std::map<mlir::Operation*, int> For_loop_level;
  std::map<mlir::Block*, int> loop_block_level;
  SmallVector<mlir::Operation*> OpsOutsideFor;
  kernel->walk([&](mlir::Operation* op)
  {
    if(isa<affine::AffineForOp>(op)){
      int Innermost = 1;
      op->walk([&](affine::AffineForOp temp_forop)
      { 
        if(For_loop_level.count(temp_forop) == 0 && temp_forop != dyn_cast<affine::AffineForOp>(op)){ // Don't count the scf::For itself
          Innermost = 0;
        // llvm::errs() << "Not innermost"  <<std::endl;    
        } 
      });
      if (Innermost)
      {
        For_loop_level[op] = level;
        level++;
      }
      if(isa<ADORA::KernelOp>(op->getParentOp()))
        OpsOutsideFor.push_back(op);
      // for_region.viewGraph();
    }
    // scf::ForOp forop;
    else if(isa<scf::ForOp>(op)){
      int Innermost = 1;
      op->walk([&](scf::ForOp temp_forop)
      { 
        if(For_loop_level.count(temp_forop) == 0 && temp_forop != dyn_cast<scf::ForOp>(op)){ // Don't count the scf::For itself
          Innermost = 0;
        // llvm::errs() << "Not innermost"  <<std::endl;    
        } 
      });
      if (Innermost)
      {
        For_loop_level[op] = level;
        level++;
      }
      // for_region.viewGraph();
    }
    else if((isa<affine::AffineLoadOp>(op) 
        || isa<affine::AffineStoreOp>(op)
        || isa<arith::AddFOp>(op)
        || isa<arith::AddIOp>(op)
        || isa<arith::SubFOp>(op)
        || isa<arith::SubIOp>(op)) 
        && isa<ADORA::KernelOp>(op->getParentOp())){
      OpsOutsideFor.push_back(op);
    }
  });
  level_total = level;
  // scf::ForOp scf_for;
  mlir::Operation* for_op;



  /*** Add Nodes ***/
  mlir::SmallVector<mlir::Operation*> AddedOps;
  for (level = 0; level < level_total; level++)
  {
    /// Find the scf_for body in current level
    for (auto op : For_loop_level)
    {
      if (op.second == level){
        for_op = op.first;
      }
    }
    /// this is a affine for
    if(isa<affine::AffineForOp>(for_op)){
      affine::AffineForOp affinefor = dyn_cast<affine::AffineForOp>(for_op);
      if(verbose) {errs() << "level:" << level << "\n"; affinefor.dump();}

      // mlir::Region *for_region = affinefor.getBody()->getParent();
      /// index and iter_args in scf_for
      // int block_cnt = 0;
      // for (auto itr=for_region->begin(); itr!=for_region->end(); itr++, block_cnt++)
      // {
      //   mlir::Block* for_block = &(*itr);
      //   errs() << "Block:" << for_block <<"\n";
      //   for_block->dump();
      //   // loop_block_level[for_block] = level;
      //   // CDFG->addNode(for_block, level); 
      // }
      // affinefor.walk([&](mlir::Operation *op){
      //   errs() << "Node:"; op->dump();
      // });
      // assert(block_cnt == 1 && "Region in affine.for should only contain 1 block !");
      affinefor.walk([&](mlir::Operation *op)
      { 
        if(verbose) {errs() << "Node:"; op->dump();}
        if(findElement(AddedOps, op) != -1)
          return WalkResult::advance();
        else
          AddedOps.push_back(op);

        if(op->getName().getStringRef() == "affine.for"){
          LLVMCDFGNode* node = CDFG->addNode(op); 
          node->setLoopLevel(level);
          // node->setisSCFForOp(true);
          // return WalkResult::advance();
        } 
        else if(op->getName().getStringRef() == "affine.yield"){
          LLVMCDFGNode* node = CDFG->addNode(op); 
          node->setLoopLevel(level);
          // TODO: settle this
          // return WalkResult::advance();
        } 
        else if(op->getName().getStringRef() == "affine.apply"){
          // LLVMCDFGNode* node = CDFG->addNode(op); 
          // node->setLoopLevel(level);
          // // TODO: settle this
          return WalkResult::advance();
        } 
        else if (op->getName().getStringRef() == "affine.load"){
          LLVMCDFGNode* node = CDFG->addNode(op); 
          node->setLoopLevel(level);
          affine::AffineLoadOp load = dyn_cast<affine::AffineLoadOp>(op);
          std::string linearaccess_str = LinearAccessToStr(GetLinearAccess(load, For_loop_level));
          std::string initAddr_str = std::to_string(GetInitAddr(load, For_loop_level));

          mlir::Operation* mrefop = load.getMemref().getDefiningOp();
          std::string ref_name;
          if(isa<ADORA::DataBlockLoadOp>(mrefop)){
            ADORA::DataBlockLoadOp Bload = dyn_cast<ADORA::DataBlockLoadOp>(mrefop);
            ref_name = std::string(Bload.getKernelName()) + "_" + std::string(Bload.getId());
          }
          else if(isa<ADORA::LocalMemAllocOp>(mrefop)){
            ADORA::LocalMemAllocOp BAlloc = dyn_cast<ADORA::LocalMemAllocOp>(mrefop);
            ref_name = std::string(BAlloc.getKernelName()) + "_" + std::string(BAlloc.getId());
          }
          else
            assert(0);
          // mrefop->dump();
          // llvm::errs() << mrefop->getName();
          int memrefsize = GetMemrefSize(load);
          // op->getResult(0).addAttribute("LinearAccess", b.getStringAttr(linearaccess_str));
          node->setLinearAccess(linearaccess_str);
          node->setInitAddr(initAddr_str);
          node->setMemrefSize(memrefsize);
          node->setMemrefName(ref_name);
          node->setLSaffine(true);
        }
        else if (op->getName().getStringRef() == "affine.store" ){
          LLVMCDFGNode* node = CDFG->addNode(op); 
          node->setLoopLevel(level);
          affine::AffineStoreOp store = dyn_cast<affine::AffineStoreOp>(op);
          std::string linearaccess_str = LinearAccessToStr(GetLinearAccess(store, For_loop_level));
          std::string initAddr_str = std::to_string(GetInitAddr(store, For_loop_level));
          int memrefsize = GetMemrefSize(store);
          mlir::Operation* mrefop = store.getMemref().getDefiningOp();
          std::string ref_name;
          if(isa<ADORA::DataBlockLoadOp>(mrefop)){
            ADORA::DataBlockLoadOp Bload = dyn_cast<ADORA::DataBlockLoadOp>(mrefop);
            ref_name = std::string(Bload.getKernelName()) + "_" + std::string(Bload.getId());
          }
          else if(isa<ADORA::LocalMemAllocOp>(mrefop)){
            ADORA::LocalMemAllocOp BAlloc = dyn_cast<ADORA::LocalMemAllocOp>(mrefop);
            ref_name = std::string(BAlloc.getKernelName()) + "_" + std::string(BAlloc.getId());
          }
          else{
            assert(0);
          }
          // op->getResult(0).addAttribute("LinearAccess", b.getStringAttr(linearaccess_str));
          node->setLinearAccess(linearaccess_str);
          node->setInitAddr(initAddr_str);
          node->setMemrefSize(memrefsize);
          node->setMemrefName(ref_name);
          node->setLSaffine(true);
        }
        else if(op->getName().getStringRef() == "arith.constant"){
          LLVMCDFGNode* node = CDFG->addNode(op); 
          setConstantNode(node);
        }
        else if(op->getName().getStringRef() == "arith.cmpi"
              ||op->getName().getStringRef() == "arith.cmpf"){
          LLVMCDFGNode* node = CDFG->addNode(op); 
          node->setLoopLevel(level);
          // node->setTypeName(GetCMPTypeStr(op));
          // ConvertGreaterToLess(node);
        }
        else{
          LLVMCDFGNode* node = CDFG->addNode(op); 
          node->setLoopLevel(level);
        }

        return WalkResult::advance();
      });
    }
  }
  /// Add those nodes outside for loop
  for(mlir::Operation* lsop : OpsOutsideFor){
    LLVMCDFGNode* node = CDFG->addNode(lsop); 
    node->setLoopLevel(level_total);
    if (lsop->getName().getStringRef() == "affine.load"){
      affine::AffineLoadOp load = dyn_cast<affine::AffineLoadOp>(lsop);
      std::string linearaccess_str = LinearAccessToStr(GetLinearAccess(load, For_loop_level));
      std::string initAddr_str = std::to_string(GetInitAddr(load, For_loop_level));

      mlir::Operation* mrefop = load.getMemref().getDefiningOp();
      std::string ref_name;
      if(isa<ADORA::DataBlockLoadOp>(mrefop)){
        ADORA::DataBlockLoadOp Bload = dyn_cast<ADORA::DataBlockLoadOp>(mrefop);
        ref_name = std::string(Bload.getKernelName()) + "_" + std::string(Bload.getId());
      }
      else if(isa<ADORA::LocalMemAllocOp>(mrefop)){
        ADORA::LocalMemAllocOp BAlloc = dyn_cast<ADORA::LocalMemAllocOp>(mrefop);
        ref_name = std::string(BAlloc.getKernelName()) + "_" + std::string(BAlloc.getId());
      }
      else
        assert(0);

      int memrefsize = GetMemrefSize(load);
      node->setLinearAccess(linearaccess_str);
      node->setInitAddr(initAddr_str);
      node->setMemrefSize(memrefsize);
      node->setMemrefName(ref_name);
      node->setLSaffine(true);
    }
    else if (lsop->getName().getStringRef() == "affine.store" ){
      affine::AffineStoreOp store = dyn_cast<affine::AffineStoreOp>(lsop);
      std::string linearaccess_str = LinearAccessToStr(GetLinearAccess(store, For_loop_level));
      std::string initAddr_str = std::to_string(GetInitAddr(store, For_loop_level));
      int memrefsize = GetMemrefSize(store);
      mlir::Operation* mrefop = store.getMemref().getDefiningOp();
      std::string ref_name;
      if(isa<ADORA::DataBlockLoadOp>(mrefop)){
        ADORA::DataBlockLoadOp Bload = dyn_cast<ADORA::DataBlockLoadOp>(mrefop);
        ref_name = std::string(Bload.getKernelName()) + "_" + std::string(Bload.getId());
      }
      else if(isa<ADORA::LocalMemAllocOp>(mrefop)){
        ADORA::LocalMemAllocOp BAlloc = dyn_cast<ADORA::LocalMemAllocOp>(mrefop);
        ref_name = std::string(BAlloc.getKernelName()) + "_" + std::string(BAlloc.getId());
      }
      else{
        assert(0);
      }
      node->setLinearAccess(linearaccess_str);
      node->setInitAddr(initAddr_str);
      node->setMemrefSize(memrefsize);
      node->setMemrefName(ref_name);
      node->setLSaffine(true);
    }
  }

  /*** Add Edges ***/
  for (auto nodepair : CDFG->nodes())
  {
    // int id = nodepair.first;
    LLVMCDFGNode *SuccNode = nodepair.second;

    /// Skip Loop index and arg node because it don't get operands 
    if(SuccNode->getTypeName() == "Loop index" 
            || SuccNode->getTypeName() == "Loop arg")
    { 
      if(verbose) { errs() << nodepair.first << ".Node:" << SuccNode->getTypeName() <<"\n";}
      continue;
    }

    mlir::Operation *op = SuccNode->operation();
    if(verbose) {errs() << nodepair.first << ".Node:";}
    if(verbose) {op->dump();}
    for (unsigned operand_idx = 0; operand_idx < op->getNumOperands(); operand_idx++)
    {
      LLVMCDFGNode *AnceNode = NULL;
      bool isBackEdge = false;
      mlir::Value _v = op->getOperand(operand_idx);

      int edgeidx;
      if (SuccNode->getTypeName() == "SEL")
        edgeidx = 2 - operand_idx;
      else
        edgeidx = operand_idx;
      
      if (_v.isa<BlockArgument>()) {
        /// Operands is a loop index or loop arg
        mlir::BlockArgument arg = _v.cast<BlockArgument>();
        mlir::Block * owner = arg.getOwner();
        int blk_level = loop_block_level[owner];
        
        // _v.dump();
        // _v.getType().dump();
        // owner->dump();
        mlir::Operation* parentop;

        if(_v.getType().isIndex()){
          isBackEdge = true;
          parentop = _v.getParentBlock()->getParentOp();
          if(isa<affine::AffineForOp>(parentop)){
            if(verbose) {errs() << "  parentop:" << *parentop <<"\n";}
            
            if(SuccNode->isLinearAccess()){
              continue;      
            }       
            else{
              AnceNode = CDFG->node(parentop);
              AnceNode->addOutputNode(SuccNode, isBackEdge);
              SuccNode->addInputNode(AnceNode, edgeidx, isBackEdge);
              CDFG->addEdge(AnceNode, SuccNode); //To fix: Edge Type
            }
          }
        }
        else{
          isBackEdge = true;
          parentop = _v.getParentBlock()->getParentOp();
          if(isa<affine::AffineForOp>(parentop)){
            AnceNode = CDFG->node(parentop);
            AnceNode->addOutputNode(SuccNode, isBackEdge);
            SuccNode->addInputNode(AnceNode, edgeidx, isBackEdge);
            CDFG->addEdge(AnceNode, SuccNode); //To fix: Edge Type            
          }
        }
        continue;

        // parentop = _v.getParentBlock()->getParentOp();
        // if(isa<func::FuncOp>(parentop))
        //   continue;
        
        // switch (arg.getArgNumber())
        // {
        // case 0: /// loop index
        //   AnceNode = CDFG->node_lpidx(blk_level);
        //   if(verbose) { errs() << "   Loop index,level:" << blk_level << ", " <<_v << ", owner:" << owner <<"\n"; }
        //   break;
        // case 1: /// loop-carried iteration args
        //   // AnceNode = CDFG->node_lparg(blk_level);
        //   // errs() << "  forop:" << *forop <<"\n";
        //   isBackEdge = true;
        //   parentop = _v.getParentBlock()->getParentOp();

        //   /// if parentop is not for op, just skip
        //   if(!isa<affine::AffineForOp>(parentop))
        //     continue;

        //   if(verbose) { errs() << "  parentop:" << *parentop <<"\n";}
        //   AnceNode = CDFG->node(parentop);
        //   AnceNode->addOutputNode(SuccNode, isBackEdge);
        //   SuccNode->addInputNode(AnceNode, edgeidx, isBackEdge);
        //   CDFG->addEdge(AnceNode, SuccNode); //To fix: Edge Type
        //   if(verbose) { errs() << "   Loop arg,level:" << blk_level << ", " <<_v << ", owner:" << owner <<"\n";}

        //   break;
        // default:
        //   assert( 0 && "The node of BlockArgument has not been stored into DFG !");
        //   break;
        // }
        // continue;
      }

      else{
        mlir::Operation *ance_op = op->getOperand(operand_idx).getDefiningOp();
        if(verbose) {errs() << "   Operands:";ance_op->dump();}

        AnceNode = CDFG->node(ance_op);
        if(AnceNode == NULL){ /// AnceNode is outside loop
          if(ance_op->getName().getStringRef() == "arith.constant"){
            AnceNode = CDFG->addNode(ance_op);
            setConstantNode(AnceNode);
          }
          else if(ance_op->getName().getStringRef() == "memref.get_global"){
            /// Tofix
            // AnceNode = CDFG->addNode(ance_op);
            continue;
          }
          else if(ance_op->getName().getStringRef() == "affine.apply"){
            continue;
          }
          else if(ance_op->getName().getStringRef() == "ADORA.BlockLoad"
                ||ance_op->getName().getStringRef() == "ADORA.LocalMemAlloc"){
            if(SuccNode->isLinearAccess())
              continue;
            else 
              assert(0); /// Todo: fix this
          }
          //   /// Extract Accumulation Operations
          //   // assert(isa<Affine::YieldOp>(op));
          //   AffineForOp ancefor = dyn_cast<AffineForOp>(ance_op);
          //   ValueRange YieldOperands =
          //       dyn_cast<AffineYieldOp>(ancefor.getBody()->getTerminator()).getOperands();
          //   assert(YieldOperands.size() == 1);
          //   for(mlir::Value Operand : YieldOperands){
          //     llvm::errs() << Operand <<"\n";
          //     mlir::Operation* OperandOp = Operand.getDefiningOp();
          //     if(   OperandOp->getName().getStringRef() == "arith.add" 
          //         || OperandOp->getName().getStringRef() == "arith.addf" 
          //         || OperandOp->getName().getStringRef() == "arith.mul"
          //         || OperandOp->getName().getStringRef() == "arith.mulf")
          //     {
          //       LLVMCDFGNode *ACCNode = CDFG->node(OperandOp);
          //       /// Set this node to be acc
          //       AnceNode = ACCNode;
          //       // AnceNode->SetBondingAccNode()
          //     }
          //     else if( OperandOp->getName().getStringRef() == "affine.for" ){
          //       /// operand is forop
          //       AffineYieldOp OldestAnceYield = getOldestAncestorYieldOp(dyn_cast<AffineForOp>(OperandOp));
          //       llvm::errs() << OldestAnceYield <<"\n";
          //       ValueRange OldestYieldOperands =
          //           dyn_cast<AffineYieldOp>(dyn_cast<AffineForOp>(OperandOp).getBody()->getTerminator()).getOperands();
          //       assert(OldestYieldOperands.size() == 1);
          //       mlir::Operation* OldestOperandOp = OldestYieldOperands[0].getDefiningOp();
          //       llvm::errs() << *OldestOperandOp <<"\n";
          //       LLVMCDFGNode *ACCNode = CDFG->node(OldestOperandOp);
          //     }
          //     else{
          //       assert(0 && "Unsupported Accumulation Type.(Support: ACC/FACC/MULACC/FMULACC)");
          //     }
          //   }
          // }
        }
      }

      AnceNode->addOutputNode(SuccNode, isBackEdge);
      SuccNode->addInputNode(AnceNode, edgeidx, isBackEdge);
      CDFG->addEdge(AnceNode, SuccNode); //To fix: Edge Type
    }
  }
  if(verbose) { CDFG->CDFGtoDOT(CDFG->name_str()+"_0_CDFG.dot");}

  ////////////////////////
  /// Extract Accumulation
  ////////////////////////
  HandleSelfCycle(CDFG, verbose);
  if(verbose) { CDFG->CDFGtoDOT(CDFG->name_str()+"_2_CDFG.dot");}
  ////////////////////////
  /// End of extracting acc op
  ////////////////////////


  ////////////////////////
  /// Handle compare node
  ////////////////////////
  auto result = HandlCompareNode(CDFG, verbose);
  if(!result) return false;

  ////////////////////////
  /// Remove redundant nodes: bitcast for trunc
  ////////////////////////
  bool removing = true;
  while(removing){
    removing = false;
    auto nodes = CDFG->nodes();
    for(auto &elem : nodes){
      // int node_id = elem.first;
      LLVMCDFGNode* node = elem.second;
      if(node->getTypeName() == "bitcast"){
        assert(node->inputNodes().size() == 1);
        LLVMCDFGNode* AnceNode = node->getInputPort(0);
        for(LLVMCDFGNode* output : node->outputNodes())
        {
          int edgeidx = output->getInputIdx(node);
          bool isbackedge = output->isInputBackEdge(node);
          output->addInputNode(AnceNode,  edgeidx, isbackedge);
          AnceNode->addOutputNode(output, isbackedge);
          CDFG->addEdge(AnceNode, output); //To fix: Edge Type     
        }
        CDFG->delNode(node);
        removing = 1;
      }
      else if(node->getTypeName() == "for"){
        ////// for to input/output
        const std::vector<LLVMCDFGNode *>& sinknodes = node->outputNodes();
        // for(LLVMCDFGNode* sinknode : sinknodes){
        //   if( ( sinknode->getTypeName()=="Input" || sinknode->getTypeName()=="INPUT"
        //       ||sinknode->getTypeName()=="Output" || sinknode->getTypeName()=="OUTPUT" )
        //     && sinknode->isLSaffine() ) {
        //       CDFG->delEdge(CDFG->edge(node, sinknode));
        //       node->delOutputNode(sinknode);
        //       sinknode->delInputNode(node); 
        //       LLVM_DEBUG( llvm::errs()<< node->inputNodes().size()<< " " << node->outputNodes().size()<< " ");
        //   }
        // }
        // LLVM_DEBUG( llvm::errs()<< node->inputNodes().size()<< " " << node->outputNodes().size()<< " ");
        // if(verbose) { CDFG->CDFGtoDOT(CDFG->name_str()+"_3_CDFG.dot");}
        assert(node->inputNodes().size() == 0 && node->outputNodes().size() == 0);
        CDFG->delNode(node);
        removing = 1;
      }
      else if(node->getTypeName() == "truncf"){
        assert(node->inputNodes().size() == 1 && node->outputNodes().size() == 0);
        CDFG->delNode(node);
        removing = 1;
      }
      else if(node->getTypeName() == "CONST"){
        if(node->inputNodes().size() == 0 && node->outputNodes().size() == 0){
          // Useless constant 
          CDFG->delNode(node);
          removing = 1;
        }  
        else if(node->outputNodes().size() != 0){
          if(dyn_cast<arith::ConstantOp>(node->operation()).getValue().getType().isIndex()){
            int i = 0;
            for(i = 0; i < node->outputNodes().size(); i++){
              LLVMCDFGNode* output = node->outputNodes()[i];
              if(verbose) {output->operation()->dump();}
              if((output->getTypeName()=="Input" || output->getTypeName()=="INPUT")
              &&(output->isLinearAccess()))
                continue;
              else
                break;
            }
            if(i == node->outputNodes().size()){
              /// All output node is linear access input, constant node can be removed
              CDFG->delNode(node);
              removing = 1;
            }
          }
        }
      }
    }    
  }

  return true;
}


LLVMCDFG* mlir::ADORA::generateCDFGfromKernel(LLVMCDFG* &CDFG, ADORA::KernelOp kernel, bool verbose){
  /// remove truncf op
  RemoveConstantTruncF(kernel);
  ADORA::KernelOp kernel_clone = kernel.clone();
  // kernel_clone.dump();
  kernel.getOperation()->getBlock()->push_back(kernel_clone);
  kernel_clone->moveBefore(kernel);

  /// Hoist load store op
  /// FIX: We should judge whether to use hoist
  HoistLoadStoreInKernelOp(kernel); 
  if(verbose) kernel.dump();

  /// For every loop carried value, find their accumulation mode. Move initial value computing to outer most level.
  MoveLoopCarriedInitailValue(kernel);
  if(verbose) kernel.dump();

  /// For accumulation chain, move accumulation operation to the last using commutative law of addition/multiplication
  MoveAccumulationToLast(kernel);
  if(verbose) kernel.dump();

  /// insert ISEL operator for loop carried value(not acc)
  InsertIselForLoopCarry(kernel, verbose);
  if(verbose) kernel.dump();

  /// Generate
  if(generateCDFGfromKernelAfterOptimization(CDFG, kernel, verbose)){
    kernel_clone.erase();
  }
  else{
    if(verbose) llvm::errs() << "[Mion] CDFG generation failed. Try again with no opt.\n";
    if(verbose) kernel.dump();
    if(verbose) kernel_clone.dump();
    LLVMCDFG* NewCDFG = new LLVMCDFG(CDFG->name_str(), CDFG->getOpNameFilePath());
    delete CDFG;
    CDFG = NewCDFG;
    if(generateCDFGfromKernelAfterOptimization(CDFG, kernel_clone, verbose)){
      kernel.erase();
    }
    else{
      llvm::errs() << "[ERROR] CDFG generation failed again. Abort.\n";
      abort();
    }
  }
  
  return CDFG;
}

void ADORALoopCdfgGenPass::runOnOperation()
{
  mlir::Operation *m = getOperation();
  llvm::errs() << "[test] kernel! " ; m->dump() ;

  /// Get function name
  func::FuncOp Func;
  unsigned cnt = 0;
  for (auto FuncOp : getOperation().getOps<func::FuncOp>())
  {
    cnt++;
    Func = FuncOp;
  }
  assert(cnt == 1 && "There should be only 1 topFunc in IR Module.");
  std::string funcname = Func.getSymName().str();

  // Get loop level from scf ForOP
  /// Generating DFG
  std::string GeneralOpNameFile_str;
  if (GeneralOpNameFile == nullptr) {
    std::cerr << "Environment variable \" GENERAL_OP_NAME_ENV \" is not set." << std::endl;
    GeneralOpNameFile_str = "/home/jhlou/CGRVOPT/cgra-opt/lib/DFG/Documents/GeneralOpName.txt";
    std::cerr << "Using \" GENERAL_OP_NAME_ENV \" = \"/home/jhlou/CGRVOPT/cgra-opt/lib/DFG/Documents/GeneralOpName.txt\"" << std::endl;
  }
  else
    GeneralOpNameFile_str = GeneralOpNameFile;
  // LLVMCDFG *CDFG = new LLVMCDFG(funcname, GeneralOpNameFile_str);

  // ADORA::KernelOp kernel;
  // OpBuilder b(m);
  // m->walk([&](ADORA::KernelOp k){
  //   kernel = k;
  //   WalkResult::interrupt();
  // });
  int kernel_cnt = 0;
  m->walk([&](ADORA::KernelOp kernel) {
    std::string kernelName = kernel.getKernelName();
    if(kernelName.empty()){
      kernelName = "kernel_" + std::to_string(kernel_cnt);
    }
    LLVMCDFG *CDFG = new LLVMCDFG(kernelName, GeneralOpNameFile_str);
    generateCDFGfromKernel(CDFG, kernel, /*verbose=*/false);
    CDFG->CDFGtoDOT(kernelName + "_CDFG.dot");
    kernel_cnt++;
  });   

  // generateCDFGfromKernel(CDFG, kernel);

  // CDFG->CDFGtoDOT(CDFG->name_str()+"_CDFG.dot");
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::ADORA::createADORALoopCdfgGenPass()
{
  return std::make_unique<ADORALoopCdfgGenPass>();
}
