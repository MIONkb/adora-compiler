//===-- Passes.td - ADORATensor pass definition file --------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines the ADORATensor Passes
//
//===----------------------------------------------------------------------===//
#ifndef ADORATENSOR_DIALECT_PASSES
#define ADORATENSOR_DIALECT_PASSES
include "ADORA/Dialect/ADORATensor/IR/ADORATensorBase.td"
include "mlir/IR/BuiltinDialect.td"
include "ADORA/Dialect/ADORA/IR/ADORABase.td"
include "mlir/Pass/PassBase.td"

def LinalgToSystolicGEMMPass : Pass<"adora-convert-linalg-to-systolic-gemm", "ModuleOp"> {
  let summary = "Transforms Linalg dialect operators into optimized systolic GEMM implementations. Following operators are included: matmul, conv2d.";
  let description = [{"
    This pass converts specific Linalg operations, such as 
    matrix multiplication and 2D convolution, into systolic 
    array implementations. 
  "}];
  let constructor = "createLinalgToSystolicGEMMPass()";
  let dependentDialects = [
    "affine::AffineDialect", "linalg::LinalgDialect", "memref::MemRefDialect", "ADORA::ADORADialect", "ADORA::ADORATensor::ADORATensorDialect"];
}

#endif 
