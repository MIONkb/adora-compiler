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

def ADORAGemmOpStrategyDecisionPass
    : Pass<"adora-gemm-op-strategy-decision", "mlir::ModuleOp"> {
  let summary = "Decide dataflow strategy for ADORA GEMM ops and annotate / rewrite CDFG";
  let description = [{
    This pass analyzes ADORA GEMM-like tensor ops, selects a dataflow strategy
    (e.g., IS/WS/OS), and materializes the decision by annotating ops or
    rewriting into a Control-Data Flow Graph (CDFG) form.
  }];
  let constructor = "createADORAGemmOpStrategyDecisionPass()";

  let dependentDialects = [
    "mlir::ADORA::ADORADialect",
    "mlir::ADORA::ADORATensor::ADORATensorDialect",
  ];

let options = [
    Option<"_adg_fn", "adg-fn", "std::string",
           /*default=*/"\"-\"",
           "ADG filename (or directly set PE numbers)">,

    Option<"_n_pes", "pe-num", "int",
           /*default=*/"160",
           "Number of PEs (Processing Elements)">,

    Option<"_n_iobs", "iob-num", "int",
           /*default=*/"32",
           "Number of IOBs (Input/Output Blocks)">,

    Option<"_sram_bank_kb_size", "bank-size", "int",
           /*default=*/"8",
           "single bank's size(KB)">,
           
    Option<"_bus_byte_band_width", "bus-bandwidth", "int",
           /*default=*/"8",
           "Bandwidth of bus (bytes)">,

    Option<"_stationary_kind", "stationary-tensor", "std::string",
           /*default=*/"\"-\"",
           "Manually set stationary kind (inputstationary/weightstationary/outputstationary, default: automatic exploration)">,
];

}

def ADORATensorOpCdfgGenPass : Pass<"adora-gen-tensor-op-cdfg", "ModuleOp"> {
  let summary = "Generate cdfg of ADORA from a ADORA tensor op";
  let constructor = "createADORATensorOpCdfgGenPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "::mlir::ADORA::ADORADialect",
    "::mlir::scf::SCFDialect",
    "::mlir::affine::AffineDialect"
  ];
}
#endif 
