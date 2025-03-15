//===- ADORADialect.cpp - MLIR Dialect for ADORA Kernels implementation -------===//
//===----------------------------------------------------------------------===//
//
// This file implements the ADORA kernel-related dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "RAAA/Dialect/ADORA/IR/ADORA.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace mlir::ADORA;

#include "RAAA/Dialect/ADORA/IR/ADORAOpsDialect.cpp.inc"

void ADORADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RAAA/Dialect/ADORA/IR/ADORAOps.cpp.inc"
      >();
}