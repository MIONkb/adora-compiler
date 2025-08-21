// SPDX-License-Identifier: Apache-2.0

//===- ShapeInferenceInterface.td - ShapeInference Interface -*- tablegen -===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Defines the operations of the Shape Inference Op Interface.
//
//===----------------------------------------------------------------------===//

#ifdef ADORATENSOR_CLONE_FROM_OTHER_DIALECT_INTERFACE_TD
#else
#define ADORATENSOR_CLONE_FROM_OTHER_DIALECT_INTERFACE_TD

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def CloneFromOtherDialectOpInterface : OpInterface<"CloneFromOtherDialectOpInterface"> {
  let description = [{
    
  }];

  let methods = [
    InterfaceMethod<
      /* doc string */
      "Clone this op from another dialect with a custom builder",
      /* return type */
      "mlir::Operation *",
      /* method name */
      "BuildFromOtherDialect",
      /* arguments */
      (ins "mlir::OpBuilder &":$builder,
           "mlir::Operation *":$otherOp)
    >,
    
    InterfaceMethod<
      /* doc string */
      "",
      /* return type */
      "unsigned ",
      /* method name */
      "getExpectedNumOperands",
      /* arguments */
      (ins)
    >
  ];
  
  let cppNamespace = "::mlir::ADORA::ADORATensor";
}


#endif // SHAPE_INFERENCE_OP_INTERFACE
