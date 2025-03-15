
/****
 * 
 * Add by Ljh 20230317
 * 
*/
#include "../inc/llvm_cdfg_node.h"
#include "../inc/mlir_cdfg_node.h"

LLVMCDFGNode::LLVMCDFGNode(mlir::Operation *op, std::string TypeName, LLVMCDFG *parent)
{
    _parent = parent;
    setTypeName(TypeName);
    setOperation(op);
}

LLVMCDFGNode::LLVMCDFGNode(mlir::Block* block, int level, std::string TypeName, LLVMCDFG *parent)
{
    _parent = parent;
    setLoopLevel(level);
    setTypeName(TypeName);
    setBlockPtr(block);
}

void LLVMCDFGNode::setOperation(mlir::Operation *op)
{ 
    _operation = op; 
    // if(dyn_cast<PHINode>(ins)){
    //     return;
    // }
    // // may have multiple constant operand, create CONST node separately
    // // for(int i = 0; i < ins->getNumOperands(); i++){ 
    // //     if(ConstantInt *CI = dyn_cast<ConstantInt>(ins->getOperand(i))){
    // //         setConstVal(CI->getSExtValue());
    // //         break; // only one constant
    // //     }
    // // }
    // if(dyn_cast<LoadInst>(ins)){
    //     setDataBits(ins->getType()->getPrimitiveSizeInBits());
    //     Value *address_of_load;
    //     if(auto GEP = dyn_cast<GetElementPtrInst>(ins->getOperand(0))){
    //         address_of_load = GEP->getOperand(0);
    //     }else if(auto GEP = dyn_cast<ConstantExpr>(ins->getOperand(0))){
    //         address_of_load = GEP->getOperand(0);                            
    //     }else{
    //         address_of_load = ins->getOperand(0);
    //     }
    //     setLSaddress(address_of_load);
    // }else if(StoreInst *SI = dyn_cast<StoreInst>(ins)){
    //     setDataBits(SI->getValueOperand()->getType()->getPrimitiveSizeInBits());
    //     Value *address_of_store;
    //     if(auto GEP = dyn_cast<GetElementPtrInst>(ins->getOperand(1))){
    //         address_of_store = GEP->getOperand(0);
    //     }else if(auto GEP = dyn_cast<ConstantExpr>(ins->getOperand(1))){
    //         address_of_store = GEP->getOperand(0);                            
    //     }else{
    //         address_of_store = ins->getOperand(1);
    //     }
    //     setLSaddress(address_of_store);
    // }else if(dyn_cast<GetElementPtrInst>(ins)){
    //     setDataBits(32);
    // }else if(dyn_cast<AllocaInst>(ins)){
    //     setDataBits(32);
    // }else if(dyn_cast<PointerType>(ins->getType())){
    //     setDataBits(32);
    // }
}