//===- Utility.cpp - Some utility tools for ADORATensor -----------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Builders.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Utility/Utility.h"
#include <iostream>
// #include <filesystem>
// #include <fstream>
// #include <regex>

using namespace mlir;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;

func::FuncOp mlir::ADORA::ADORATensor::
    ConvertMatmulToFunc(mlir::linalg::MatmulOp op, llvm::SetVector<mlir::Value> &operands, std::string FnName){
  return ConvertOpTtoFunc<mlir::linalg::MatmulOp>(op, operands, FnName);
}