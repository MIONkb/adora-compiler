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

#ifndef ADORATENSOR_CLONE_FROM_OTHER_DIALECT_INTERFACE_H
#define ADORATENSOR_CLONE_FROM_OTHER_DIALECT_INTERFACE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"

/// Include the auto-generated declarations.
#include "RAAA/Dialect/ADORATensor/Interface/ShapeInferenceOpInterface.h.inc"
#endif
