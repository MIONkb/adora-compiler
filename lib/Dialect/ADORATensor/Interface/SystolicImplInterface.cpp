//===---- SystolicImplInterface.cpp - Definition for SystolicImplInterface ---===//

// =============================================================================
//
// This file contains the implementations of the Systolic Implement Interface
// defined in SystolicImplInterface.td.
//
//===----------------------------------------------------------------------===//

#include "RAAA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"

namespace mlir {

/// Include the auto-generated declarations.
#include "RAAA/Dialect/ADORATensor/Interface/SystolicImplInterface.cpp.inc"

namespace ADORA {
namespace ADORATensor {
    
StringRef getMethodStrRef(MatMulStrategy method) {
  switch (method) {
    case MatMulStrategy::WeightStationary:
      return "WeightStationary";
      break;
    case MatMulStrategy::InputStationary:
      return "InputStationary";
      break;
    case MatMulStrategy::OutputStationary:
      return "OutputStationary";
      break;
    }
}
} // end namespace ADORATensor
} // end namespace ADORA


} // end namespace mlir
