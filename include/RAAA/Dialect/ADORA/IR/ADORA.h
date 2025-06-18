//===- Test.h - Test dialect --------------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef CGRAOPT_DIALECT_ADORA_IR_Test_H_
#define CGRAOPT_DIALECT_ADORA_IR_Test_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// #include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"

// #include "llvm/ADT/SmallSet.h" /// use std::unordered_set instead of std::list
#include<set>
//===----------------------------------------------------------------------===//
// Test Dialect
//===----------------------------------------------------------------------===//

#include "RAAA/Dialect/ADORA/IR/ADORAOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Test Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "RAAA/Dialect/ADORA/IR/ADORAOps.h.inc"

#include "RAAA/Dialect/ADORA/IR/ADORAOpsTypes.h.inc"
//===----------------------------------------------------------------------===//
// ADORA Dialect Helpers
//===----------------------------------------------------------------------===//

namespace mlir {
namespace ADORA {
///////////////
/// AdjustMemoryFootprint.cpp
///////////////
std::optional<int64_t> getSingleMemrefAccessSpace(::mlir::affine::AffineForOp forOp);

///////////////
/// HoistLoadStore.cpp
///////////////
enum class PositionRelationInLoop { SameLevel, LhsOuter, RhsOuter, NotInSameLoopNest}; 
PositionRelationInLoop getPositionRelationship(Operation* lhs, Operation* rhs);
template <typename LoadOrStoreT, typename OpToWalkT>
  SmallVector<LoadOrStoreT,  4> GetAllHoistOp(OpToWalkT op_tocheck);
std::optional<mlir::affine::AffineForOp> MoveLoadStorePairOut(::mlir::affine::AffineLoadOp loadop, ::mlir::affine::AffineStoreOp storeop);

///////////////
/// DFGgen.cpp
///////////////
// void HoistLoadStoreInKernelOp(ADORA::KernelOp kernel);

///////////////
/// Utilities.cpp
///////////////
// Define supported affine operations on CGRA
typedef std::set<StringRef> OpTable;

mlir::Operation* eraseKernel(::mlir::func::FuncOp& TopFunc, ADORA::KernelOp& Kernel);
LogicalResult SpecifiedAffineFortoKernel(::mlir::affine::AffineForOp& forOp);
LogicalResult SpecifiedAffineFortoKernel(::mlir::affine::AffineForOp& forOp, std::string kernel_name);
AffineExpr getConstPartofAffineExpr(AffineExpr& expr);
signed MultiplicatorOfDim(const AffineExpr& expr, const unsigned dim);
// void removeUnusedRegionArgs(Region &region);
void eliminateUnusedIndices(Operation *op);
::llvm::SmallDenseMap<mlir::Value, unsigned> getOperandInRank(Operation *op, unsigned rank);
SmallVector<mlir::Operation*>  getAllUsesInRegion(const mlir::Value beused, ::mlir::Region* region);
SmallVector<mlir::Operation*>  getAllUsesInBlock(const mlir::Value beused, ::mlir::Block* block);

mlir::Operation* GetTheSourceOperationOfBlockStore(ADORA::DataBlockStoreOp store);

void ResetIndexOfBlockAccessOpInFunc(func::FuncOp& func);
inline bool opIsContainedByKernel(mlir::Operation* op);

void LLVM_ATTRIBUTE_UNUSED 
 simplifyMapWithOperands(AffineMap &map, ArrayRef<mlir::Value> operands);

///// following 3 functions are defined to simplify AffineApplyOp
void simplifyConstantAffineApplyOpsInRegion(::mlir::Region& region);
void simplifyAddAffineApplyOpsInRegionButOutOfKernel(::mlir::Region& region);
void simplifyLoadAndStoreOpsInRegion(::mlir::Region& region);

///// following 4 functions are defined to help extract kernel function
// bool isSinkingBeneficiary(Operation *op);
// static bool extractBeneficiaryOps(Operation *op, llvm::SetVector<Value> existingDependencies,
//       llvm::SetVector<Operation *> &beneficiaryOps, llvm::SmallPtrSetImpl<Value> &availableValues);
LogicalResult sinkOperationsIntoKernelOp(ADORA::KernelOp kernelOp);
ADORA::KernelOp getSingleKernelFromFunc(func::FuncOp func);
ADORA::KernelOp getKernelFromCopiedModule(ModuleOp ModuleOp, ADORA::KernelOp kernel);

func::FuncOp GenKernelFunc(ADORA::KernelOp KernelOp, llvm::SetVector<Value> &operands);
bool IsIterationSpaceSupported(mlir::affine::AffineForOp &forOp);

bool LoadStoreSameMemAddr(::mlir::affine::AffineLoadOp loadop, ::mlir::affine::AffineStoreOp storeop);
SmallVector<int> getOperandDimensionsInMap(const int dim, const ::mlir::AffineMap map);
unsigned const getInstanceNumFromADG  (const std::string& CGRAadg, const std::string& instype_to_count);

ADORA::IselOp ReplaceValueWithNewIselOp(OpBuilder b, Location loc, mlir::Value value);
ADORA::IselOp ReplaceLoopCarryValueWithNewIselOp(affine::AffineForOp& forop, int IterRegionOperandIdx);

//===----------------------------------------------------------------------===//
// A templated find func for smallvector
//===----------------------------------------------------------------------===//
template <typename T, unsigned N>
inline int findElement(const llvm::SmallVector<T, N>& vec, const T& elem) {
  for (unsigned i = 0; i < vec.size(); ++i) {
    if (vec[i] == elem) {
      return i;
    }
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// A templated find func for value range
//===----------------------------------------------------------------------===//
inline mlir::Value findElement(const ValueRange vec, const mlir::Value& elem) {
  for (ValueRange::iterator itr = vec.begin(); itr != vec.end(); ++itr) {
    if (*itr == elem) {
      return *itr;
    }
  }
  
  return NULL;
}

template <typename T>
inline SmallVector<T> SetMergeForVector(const llvm::SmallVector<T>& v1, const llvm::SmallVector<T>& v2){
  SmallVector<T> v;
  for(auto e : v1){
    if(findElement(v, e) == -1){
      v.push_back(e);
    }
  }
  for(auto e : v2){
    if(findElement(v, e) == -1){
      v.push_back(e);
    }
  }
  return v;
}

} // namespace ADORA
} // namespace mlir

#endif //CGRAOPT_DIALECT_ADORA_IR_Test_H_
