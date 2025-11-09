//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef ADORATENSOR_DIALECT_UTILITY_H_
#define ADORATENSOR_DIALECT_UTILITY_H_

#include "mlir/Pass/Pass.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"

namespace mlir {
namespace ADORA {
namespace ADORATensor{


/// @brief ToFunc
/// @param Op
/// @param FnName
/// @param operands
/// @return
template <typename OpT>
static func::FuncOp ConvertOpTtoFunc(OpT op, llvm::SetVector<mlir::Value> &operands, std::string FnName)
{
  Location loc = op.getLoc();
  // Create a builder with no insertion point, insertion will happen separately
  // due to symbol table manipulation
  OpBuilder builder(op.getContext());
  // Identify uses from values defined outside of the scope of the launch
  // operation.
  // getUsedValuesDefinedAbove(KernelOpBody, operands);

  // Create the func.func operation.
  SmallVector<mlir::Type, 4> OperandTypes, ResultTypes;
  OperandTypes.reserve(operands.size());
  for (mlir::Value operand : operands)
  {
    // errs()  << "  operands:"; operand.dump();
    OperandTypes.push_back(operand.getType());
  }


  for (mlir::Value result : op.getResults())
  {
    // errs()  << "  operands:"; operand.dump();
    ResultTypes.push_back(result.getType());
  }

  mlir::FunctionType type =
      mlir::FunctionType::get(op.getContext(), OperandTypes, ResultTypes);
  func::FuncOp Func = builder.create<func::FuncOp>(loc, FnName, type);
//   LLVM_DEBUG(llvm::errs() << "[debug] after create:\n"; Func.dump(););
  // KernelFunc->setAttr(kernelFnName, builder.getUnitAttr());
  // KernelFunc->setAttr("Kernel", builder.getUnitAttr());

  /// Pass func arguements outside of KernelOp
  Block *entryBlock = new Block;
  for (mlir::Type argTy : type.getInputs())
  {
    entryBlock->addArgument(argTy, loc);
  }

  Func.getBody().getBlocks().push_back(entryBlock);

  mlir::IRMapping mapping;
  // // Block &entryBlock = KernelFunc.getBody().front();
  for (unsigned index = 0; index < operands.size(); index++)
  {
    // errs()  << "  operands:" << operands[index] <<"\n";
    mapping.map(operands[index], entryBlock->getArgument(index));
  }
  mlir::Operation* newop = op.getOperation()->clone(mapping);
  entryBlock->push_back(newop);
  entryBlock->push_back(builder.create<func::ReturnOp>(newop->getLoc(), dyn_cast<OpT>(newop).getResultTensors()));

  // Block &KernelOpEntry = KernelOpBody.front();
  // Block *clonedKernelOpEntry = mapping.lookup(&KernelOpEntry);
  // builder.setInsertionPointToEnd(entryBlock);
  // builder.create<cf::BranchOp>(loc, clonedKernelOpEntry);

  // KernelFunc.walk([](ADORA::TerminatorOp op) {
  //   OpBuilder replacer(op);
  //   replacer.create<func::ReturnOp>(op.getLoc());
  //   op.erase(); 
  // });
  func::CallOp callop = builder.create<func::CallOp>(op.getLoc(), Func, operands.getArrayRef());
  op.getOperation()->getBlock()->push_back(callop);
  callop.getOperation()->moveBefore(op.getOperation());
  op.getOperation()->replaceAllUsesWith(callop);
  
//   LLVM_DEBUG(llvm::errs() << "[debug] func:\n"; Func.dump(););
//   LLVM_DEBUG(llvm::errs() << "[debug] callop:\n"; callop.dump(););
  return Func;
}

} // namespace ADORATensor
} // namespace ADORA
} // namespace mlir

#endif // ADORATENSOR_DIALECT_UTILITY_H_
