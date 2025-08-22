//===-- ADORATensorOps.td - ADORA dialect operation definitions ------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines some operations of the ADORATensor dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ADORATensor_OPS
#define ADORATensor_OPS

include "RAAA/Dialect/ADORATensor/IR/ADORATensorBase.td"

include "mlir/Pass/PassBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Dialect/Linalg/IR/LinalgBase.td"
// include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
// include "mlir/Dialect/Arith/IR/ArithBase.td"
// include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"

// include "mlir/Dialect/Affine/IR/AffineOps.td"

// include "mlir/Dialect/Func/IR/FuncOps.td"
// include "mlir/Dialect/Func/IR/FuncOps.h"
include "RAAA/Dialect/ADORATensor/Interface/ShapeInferenceOpInterface.td"
// include "RAAA/Dialect/ADORATensor/Interface/CloneFromOtherDialectOpInterface.td"
include "RAAA/Dialect/ADORATensor/Interface/SystolicImplInterface.td"

// include "mlir/Dialect/MemRef/IR/MemRefOps.td"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
//===----------------------------------------------------------------------===//
// ADORATensor Dialect operations.
//===----------------------------------------------------------------------===//
class ADORATensor_Op<string mnemonic, list<Trait> traits = []> :
    Op<ADORATensor_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// Matmul op
//===----------------------------------------------------------------------===//
/// Similar to ONNX-MLIR ONNXMatMulOp

def MatMulOp : ADORATensor_Op<"matmul",
    [DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<SystolicImplInterface>]> {
      // DeclareOpInterfaceMethods<CloneFromOtherDialectOpInterface>
  let summary = "ADORA Tensor MatMul operation";
  let description = [{
    Matrix product for CGRA deployment;
  }];
  let arguments = (ins 
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, AnyMemRef]>:$A,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, AnyMemRef]>:$B);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, AnyMemRef]>:$O);
  let extraClassDeclaration = [{
    static constexpr unsigned getExpectedNumOperands() { return 2; }

    int getNumberOfOperands() {
      return 2;
    }
    int getNumberOfResults() {
      return 1;
    }
  }];
  let extraClassDefinition = [{

  }];
}

def GemmOp : ADORATensor_Op<"Gemm",
    [
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<SystolicImplInterface>]> {
      // DeclareOpInterfaceMethods<CloneFromOtherDialectOpInterface>
  let summary = "ADORA Tensor Gemm operation";
  let description = [{
    Gemm with 3 inputs for CGRA deployment;
  }];
  let arguments = (ins 
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, AnyMemRef]>:$A,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, AnyMemRef]>:$B,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, AnyMemRef]>:$C);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, AnyMemRef]>:$O);
  let extraClassDeclaration = [{
    static constexpr unsigned getExpectedNumOperands() { return 3; }

    int getNumberOfOperands() {
      return 3;
    }

    int getNumberOfResults() {
      return 1;
    }
  }];
  let extraClassDefinition = [{

  }];
}

#endif // ADORATensor_OPS
