/*
 * SPDX-License-Identifier: Apache-2.0
 */
/// Copy from ONNX-MLIR ShapeInferenceOpInterface.hpp
//===---- ShapeInferenceOpInterface.hpp - Definition for ShapeInference ---===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef ADORATENSOR_SYSTOLIC_IMPLEMENT_INTERFACE_H
#define ADORATENSOR_SYSTOLIC_IMPLEMENT_INTERFACE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"

/// Include the auto-generated declarations.
namespace mlir
{
  namespace ADORA
  {
    namespace ADORATensor
    {
      enum class ComputeAlgorithm
      {
        GEMM_Standard,
        Conv_Direct,
        Conv_Im2Col,
        Conv_Winograd,
        Undefine
      };

      enum class DataflowStrategy
      {
        WeightStationary,
        InputStationary,
        OutputStationary,

        Undefine
      };

      struct SystolicConfig
      {
        ComputeAlgorithm algorithm;
        DataflowStrategy dataflow;

        // DSE data
        SmallVector<int64_t> loopOrder;
        SmallVector<int64_t> tileSizes;
      };

      SystolicConfig parseSystolicConfig(Operation *op);
      StringRef getComputeAlgorithmStrRef(ComputeAlgorithm algorithm);
      ComputeAlgorithm parseAlgorithmStr(StringRef str);

      StringRef getDataflowStrategyStrRef(DataflowStrategy strategy);
      DataflowStrategy parseDataflowStr(StringRef str);

    }
  }
}

#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h.inc"

#endif