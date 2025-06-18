#include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Affine/Analysis/Utils.h"
// #include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Support/LLVM.h"
// #include "mlir/Support/FileUtilities.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/RegionUtils.h"

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Debug.h"

#include "RAAA/Dialect/ADORA/IR/ADORA.h"
#include "RAAA/Dialect/ADORA/Transforms/Passes.h"
#include "RAAA/Dialect/ADORA/Transforms/DSE.h"

using namespace llvm; // for llvm.errs()
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;

#define DEBUG_TYPE "adora-simplify-loadstore"

namespace mlir {
namespace ADORA {
//===----------------------------------------------------------------------===//
// Template functions for SimplifyLoadStore Pass
//===----------------------------------------------------------------------===//
/// @brief check whether the two accesses are accessing the same memref
///        and the same address
/// @param access0 
/// @param load1
/// @return 
template <typename Access0_T, typename Access1_T>
bool TwoAccessSameMemAddr(Access0_T access0, Access1_T access1)
{
  mlir::Value Memref0 = access0.getMemref();
  AffineMapAttr MapAttr0 = access0.getAffineMapAttr();
  Operation::operand_range Indices0 = access0.getIndices();
  Operation* MemrefOp0 = Memref0.getDefiningOp();
  
  mlir::Value Memref1 = access1.getMemref();
  AffineMapAttr MapAttr1 = access1.getAffineMapAttr();
  Operation::operand_range Indices1 = access1.getIndices();
  Operation* MemrefOp1 = Memref1.getDefiningOp();

  // mlir::Value storeValue = storeop.getValue();
  // if(!isa<BlockArgument>(storeValue) && isa<arith::ConstantOp>(storeValue.getDefiningOp()))
  //   return false;

  // llvm::errs() << "[info] loadop: " << loadop << "\n";   
  // llvm::errs() << "[info] loadMemref: " << loadMemref << "\n";   
  // llvm::errs() << "[info] loadMapAttr: " << loadMapAttr << "\n";   

  // llvm::errs() << "[info] storeop: " << storeop << "\n"; 
  // llvm::errs() << "[info] storeMemref: " << storeMemref << "\n"; 
  // llvm::errs() << "[info] storeMapAttr: " << storeMapAttr << "\n"; 

  
  if(Memref0 == Memref1 && MapAttr0 == MapAttr1
            && Indices0.size() == Indices1.size())
  {
    for(unsigned i = 0; i < Indices0.size(); i++){
      if(Indices0[i] != Indices1[i])
        return false;
    }
    return true;
  }
  else if(
    !isa<BlockArgument>(Memref0) && !isa<BlockArgument>(Memref1)
    && isa<ADORA::DataBlockLoadOp>(MemrefOp0) && isa<ADORA::DataBlockLoadOp>(MemrefOp1)
    && dyn_cast<ADORA::DataBlockLoadOp>(MemrefOp0).getOriginalMemref() 
    == dyn_cast<ADORA::DataBlockLoadOp>(MemrefOp1).getOriginalMemref())
  {
    for(unsigned i = 0; i < Indices0.size(); i++){
      if(Indices0[i] != Indices1[i])
        return false;
    }
    return true;      
  }
  
  else if(!isa<BlockArgument>(Memref0) && !isa<BlockArgument>(Memref1)
      &&isa<ADORA::DataBlockLoadOp>(MemrefOp0) 
      &&isa<ADORA::LocalMemAllocOp>(MemrefOp1)){
    ADORA::LocalMemAllocOp AllocOp = dyn_cast<ADORA::LocalMemAllocOp>(MemrefOp1);
    SmallVector<mlir::Operation*> Consumers = 
      getAllUsesInBlock(AllocOp, AllocOp.getOperation()->getBlock());
    // AllocOp.getOperation()->getBlock()->dump();
    // AllocOp.getOperation()->getBlock()->getRegion()->dump();
    SmallVector<ADORA::DataBlockStoreOp> BLKStoreConsumers;
    for(auto comsumer: Consumers) {
      if(isa<ADORA::DataBlockStoreOp>(comsumer))
        BLKStoreConsumers.push_back(dyn_cast<ADORA::DataBlockStoreOp>(comsumer));
    }
      // comsumer->dump();
    assert(BLKStoreConsumers.size() == 1);
    Memref1 = BLKStoreConsumers[0].getTargetMemref();

    if(dyn_cast<ADORA::DataBlockLoadOp>(MemrefOp0).getOriginalMemref() == Memref1)
    {
      for(unsigned i = 0; i < Indices0.size(); i++){
        if(Indices0[i] != Indices1[i])
          return false;
      }
      return true;      
    }
    else 
      return false;
  }
   
  else{
    return false;
  }
}


template <typename LoadOrStoreT, typename OpToWalkT>
SmallVector<LoadOrStoreT,  4> GetAllHoistOp(OpToWalkT op_to_walk){
  SmallVector<LoadOrStoreT,  4> ToHoistOps;
  op_to_walk.walk([&](LoadOrStoreT accessop)
  {
    Operation* ParentOp = accessop.getOperation()->getParentOp();
    if(ParentOp->getName().getStringRef() == AffineForOp::getOperationName() )
    { 
      AffineForOp ParentForOp = dyn_cast<AffineForOp>(*ParentOp);
      // Value memref;
      // SmallVector<Value, 4> IVs;
      for(Value index : accessop.getIndices()){
        if(index == ParentForOp.getInductionVar()){
          // This accessop op can't be hoisted because it is constrained by loop of its level. 
          return WalkResult::advance();
        }
      }
      // This accessop op can be hoisted because it is not constrained by IV of its parent fopOp. 
      ToHoistOps.push_back(accessop);
    }
    return WalkResult::advance();
  });
  
  return ToHoistOps;
}


/// @brief walk op_to_walk(affine.if or affine.for) to find whether a memref access 
///        same to Op_to_check exsits 
/// @param Op_to_check
/// @param op_to_walk
/// @return 
template <typename LoadOrStoreT, typename OpToWalkT>
bool WalkOpToCheckWhetherExistStoreTheSameMemref(LoadOrStoreT Op_to_check, OpToWalkT op_to_walk){
  mlir::Value Memref = Op_to_check.getMemref();
  // AffineMapAttr MapAttr0 = load0.getAffineMapAttr();
  // Operation::operand_range Indices0 = load0.getIndices();
  // Operation* MemrefOp = Memref.getDefiningOp();

  //// To simplify, we decide that if one store op access the same memref, then 
  ////  return true.
  auto result = op_to_walk.walk([&](AffineStoreOp accessop)->WalkResult{
    // LLVM_DEBUG(accessop.dump());

    // LLVM_DEBUG(Memref.dump());
    // LLVM_DEBUG(accessop.getMemref().dump());
    if(accessop.getMemref() == Memref){
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  if(result.wasInterrupted()){
    return true;
  }

  return false;
}

/// @brief walk op_to_walk(affine.if or affine.for) to find whether a memref access 
///        same to Op_to_check exsits 
/// @param Op_to_check
/// @param op_to_walk
/// @return 
template <typename LoadOrStoreT, typename OpToWalkT>
bool WalkOpToCheckWhetherExistLoadTheSameMemref(LoadOrStoreT Op_to_check, OpToWalkT op_to_walk){
  mlir::Value Memref = Op_to_check.getMemref();
  // AffineMapAttr MapAttr0 = load0.getAffineMapAttr();
  // Operation::operand_range Indices0 = load0.getIndices();
  // Operation* MemrefOp = Memref.getDefiningOp();

  //// To simplify, we decide that if one store op access the same memref, then 
  ////  return true.
  auto result = op_to_walk.walk([&](AffineLoadOp accessop)->WalkResult{
    if(accessop.getMemref() == Memref){
      return WalkResult::interrupt();
    }
        
    return WalkResult::advance();
  });
  if(result.wasInterrupted()){
    return true;
  }

  return false;
}

/// @brief check whether the two access operations are accessing the same memref
///        and the same address
/// @param access0
/// @param access1
/// @return 
template <typename Access0_T, typename Access1_T>
bool ConsecutiveAccessSameMemAddr(Access0_T access0, Access1_T access1){
   //// First, Check whether the two loads are consecutive loads in same block.
  mlir::Block* block = access0.getOperation()->getBlock();

  LLVM_DEBUG(llvm::errs() << "access0:" << access0 << "\n");
  LLVM_DEBUG(llvm::errs() << "access1:" << access1 << "\n");
  LLVM_DEBUG(llvm::errs() << "access0 block:\n" );
  LLVM_DEBUG(block->dump());

  bool find_access0 = false;
  for(auto iter = block->begin(); iter != block->end(); iter++){
    LLVM_DEBUG(llvm::errs() << "iter:" << *iter << "\n");
    if(!find_access0 && isa<Access0_T>(iter)){
      Access0_T access = dyn_cast<Access0_T>(iter);
      if(access0 == access){
        find_access0 = true;
      }
    }
    else if(find_access0 && isa<Access1_T>(iter)){
      Access1_T access = dyn_cast<Access1_T>(iter);
      if(access1 == access){
        ////Find the target load, Check whether the two loads access the same position.
        if(TwoAccessSameMemAddr(access0, access1)){
          return true;
        }
      }
    }
    else if(find_access0){ //// current operation is not a affine load
      if(std::is_same<Access0_T, AffineStoreOp>::value && std::is_same<Access1_T, AffineStoreOp>::value
         && isa<AffineLoadOp>(iter)){
        /// Only store-store need check intermediate load. Really?
        AffineLoadOp load = dyn_cast<AffineLoadOp>(iter);
        if(TwoAccessSameMemAddr(access0, load)){
          return false;
        }
      }
      else if(isa<AffineStoreOp>(iter)){
        AffineStoreOp store = dyn_cast<AffineStoreOp>(iter);
        if(TwoAccessSameMemAddr(access0, store)){
          return false;
        }
      }
      else if(isa<AffineIfOp>(iter)){
        AffineIfOp ifop = dyn_cast<AffineIfOp>(iter);
        if(WalkOpToCheckWhetherExistStoreTheSameMemref(access1, ifop)){
          return false;
        }
        if(std::is_same<Access0_T, AffineStoreOp>::value && std::is_same<Access1_T, AffineStoreOp>::value){
          /// for store-store case
          // auto store = dyn_cast<AffineStoreOp>(access1);
          if(WalkOpToCheckWhetherExistLoadTheSameMemref(access1, ifop))
            return false;
        }
      }
      else if(isa<AffineForOp>(iter)){
        AffineForOp forop = dyn_cast<AffineForOp>(iter);
        if(WalkOpToCheckWhetherExistStoreTheSameMemref(access1, forop)){
          return false;
        }
        if(std::is_same<Access0_T, AffineStoreOp>::value && std::is_same<Access1_T, AffineStoreOp>::value){
          // auto store = dyn_cast<AffineStoreOp>(access1);
          if(WalkOpToCheckWhetherExistLoadTheSameMemref(access1, forop))
            return false;
        }
      }
    }
  }

  return false;
}


/// Hoist Load Store Operations 
template <typename RegionOpT>
bool HoistLoadStoreOpsInOp(RegionOpT RegionOp){
  bool NoChange = 0;
  while(!NoChange){ // Keep walking the func until no change occurs in this func
    NoChange = 1;
    SmallVector<AffineLoadOp,  4> ToHoistLoads;
    SmallVector<AffineStoreOp, 4> ToHoistStores;
    /////////
    /// Step 1 : Get all loads op to be hoisted
    /////////
    ToHoistLoads = GetAllHoistOp<AffineLoadOp>(RegionOp);

    /////////
    /// Step 2 : Get all stores op to be hoisted
    /////////
    ToHoistStores = GetAllHoistOp<AffineStoreOp>(RegionOp);

    /////////
    /// Step 3 : Do hoists for load-store pairs
    /////////
    SmallVector<AffineLoadOp,  4> ToHoistLoads_copy = ToHoistLoads;
    SmallVector<AffineStoreOp, 4> ToHoistStores_copy = ToHoistStores;
    for(AffineLoadOp loadop : ToHoistLoads_copy){
      /// Check whether this load occurs with a corresponding store which have
      /// the same memref and address to access. 
      /// If so, this load-store pair 
      /// should be hoisted while construct a loop-carried variable.
      for(AffineStoreOp storeop : ToHoistStores_copy){
        // LLVM_DEBUG(llvm::errs() << "[info] loadop: " << loadop << "\n");   
        // LLVM_DEBUG(llvm::errs() << "[info] storeop: " << storeop << "\n");    
        if(ConsecutiveAccessSameMemAddr(loadop, storeop)){
          // LLVM_DEBUG(RegionOp.dump());
          if(affine::AffineForOp newforop = *mlir::ADORA::MoveLoadStorePairOut(loadop, storeop)){
            if constexpr (std::is_same<RegionOpT, affine::AffineForOp>::value) {
              //// Only when RegionOpT is affine::AffineForOp will compile following statement
              RegionOp = newforop;
            }

            // LLVM_DEBUG(RegionOp.dump());
            // Remove hoisted load/store ops from to-check vector 
            AffineLoadOp* it_ld = std::find(ToHoistLoads.begin(), ToHoistLoads.end(), loadop);
            assert(it_ld != ToHoistLoads.end());
            ToHoistLoads.erase(it_ld);
            AffineStoreOp* it_st = std::find(ToHoistStores.begin(), ToHoistStores.end(), storeop);
            assert(it_st != ToHoistStores.end());
            ToHoistStores.erase(it_st);
            NoChange = 0;
            break;
          }
        }
      }
    }
    // LLVM_DEBUG(RegionOp.dump());

    // /////////
    // /// Step 4 : Do hoists for remaining load/store /// There may be some error
    // /////////
    // for(AffineLoadOp loadop : ToHoistLoads){
    //   /// If the loadop has no corresponding store then just hoist.
    //   Operation* ParentOp = loadop.getOperation()->getParentOp();
    //   if(isa<ADORA::KernelOp>(ParentOp))
    //     continue;

    //   assert(ParentOp->getName().getStringRef() == AffineForOp::getOperationName());
    //   NoChange = 0;
    //   loadop.getOperation()->moveBefore(ParentOp);
    // }    
    // for(AffineStoreOp storeop : ToHoistStores){
    //   Operation* ParentOp = storeop.getOperation()->getParentOp();
    //   if(isa<ADORA::KernelOp>(ParentOp))
    //     continue;

    //   assert(ParentOp->getName().getStringRef() == AffineForOp::getOperationName());
    //   NoChange = 0;
    //   storeop.getOperation()->moveAfter(ParentOp);
    // }
  }

  return RegionOp;
}




} // end namespace ADORA
} // end namespace mlir