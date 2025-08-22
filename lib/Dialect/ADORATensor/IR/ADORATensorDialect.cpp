//===- ADORATensorDialect.cpp - MLIR Dialect for ADORA Tenosr implementation -------===//
//===----------------------------------------------------------------------===//
//
// This file implements high-level ADORATensor dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "RAAA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace mlir::ADORA::ADORATensor;

#include "RAAA/Dialect/ADORATensor/IR/ADORATensorOpsDialect.cpp.inc"
#include "RAAA/Dialect/ADORATensor/IR/ADORATensorOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "RAAA/Dialect/ADORATensor/IR/ADORATensorOps.cpp.inc"

void ADORATensorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RAAA/Dialect/ADORATensor/IR/ADORATensorOps.cpp.inc"
      >();
}

