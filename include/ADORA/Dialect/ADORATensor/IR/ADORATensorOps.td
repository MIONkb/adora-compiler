//===-- ADORATensorOps.td - ADORA dialect operation definitions ------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines some operations of the ADORATensor dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ADORATensor_OPS
#define ADORATensor_OPS

include "ADORA/Dialect/ADORATensor/IR/ADORATensorBase.td"

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
include "ADORA/Dialect/ADORATensor/Interface/ShapeInferenceOpInterface.td"
// include "ADORA/Dialect/ADORATensor/Interface/CloneFromOtherDialectOpInterface.td"
include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.td"

// include "mlir/Dialect/MemRef/IR/MemRefOps.td"
// #include "mlir/Dialect/Func/IR/FuncOps.h"

//===----------------------------------------------------------------------===//
// ADORATensor Type Definitions (ONNX Style Simplification)
//===----------------------------------------------------------------------===//

// Define all supported numeric element types.
def Adora_NumTypes : AnyTypeOf<[F16, F32, F64, UI32, UI64, I32, I64, BF16]>;

// Define a generic Tensor or MemRef type.
// This is a helper to avoid repeating a long list of types in every operation.
def Adora_AnyTensorOrMemRef : AnyTypeOf<[
    TensorOf<[Adora_NumTypes]>, 
    AnyMemRef
]>;

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
    Adora_AnyTensorOrMemRef:$A,
    Adora_AnyTensorOrMemRef:$B);
  let results = (outs Adora_AnyTensorOrMemRef:$O);
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

//===----------------------------------------------------------------------===//
// Gemm op
//===----------------------------------------------------------------------===//

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
    Adora_AnyTensorOrMemRef:$A,
    Adora_AnyTensorOrMemRef:$B,
    Adora_AnyTensorOrMemRef:$C
  );
  let results = (outs Adora_AnyTensorOrMemRef:$O);
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

//===----------------------------------------------------------------------===//
// Conv op
//===----------------------------------------------------------------------===//

def ConvOp : ADORATensor_Op<"Conv",
    [
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<SystolicImplInterface>]> {
  
  let summary = "ADORA Tensor Convolution operation";
  let description = [{
    Convolution operation for CGRA deployment.
  }];

  // Input: X(Input), W(Weight), B(Bias - Optional)
  let arguments = (ins 
    Adora_AnyTensorOrMemRef:$X,
    Adora_AnyTensorOrMemRef:$W,
    Optional<Adora_AnyTensorOrMemRef>:$B,

    // Key attributes for DSE modeling and lowering.
    // These attributes correspond to the standard ONNX Conv attributes.
    OptionalAttr<StrAttr>:$auto_pad,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<I64Attr>:$group,
    OptionalAttr<I64ArrayAttr>:$kernel_shape,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );

  let results = (outs Adora_AnyTensorOrMemRef:$Y);

  let extraClassDeclaration = [{
    bool hasBias() { return getB() != nullptr; }
    
    int getNumberOfOperands() {
      return hasBias() ? 3 : 2;
    }
    
    int getNumberOfResults() {
      return 1;
    }
  }];
}

#endif // ADORATensor_OPS
