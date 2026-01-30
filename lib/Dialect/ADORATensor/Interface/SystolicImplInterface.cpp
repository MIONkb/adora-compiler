//===---- SystolicImplInterface.cpp - Definition for SystolicImplInterface ---===//

// =============================================================================
//
// This file contains the implementations of the Systolic Implement Interface
// defined in SystolicImplInterface.td.
//
//===----------------------------------------------------------------------===//

#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"

namespace mlir
{

/// Include the auto-generated declarations.
#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.cpp.inc"

  namespace ADORA
  {
    namespace ADORATensor
    {
      DataflowStrategy parseDataflowStr(StringRef str)
      {
        if (str == "WeightStationary")
          return DataflowStrategy::WeightStationary;
        if (str == "InputStationary")
          return DataflowStrategy::InputStationary;
        if (str == "OutputStationary")
          return DataflowStrategy::OutputStationary;
        return DataflowStrategy::Undefine;
      }

      ComputeAlgorithm parseAlgorithmStr(StringRef str)
      {
        if (str == "GEMM_Standard")
          return ComputeAlgorithm::GEMM_Standard;
        if (str == "Conv_Direct")
          return ComputeAlgorithm::Conv_Direct;
        if (str == "Conv_Im2Col")
          return ComputeAlgorithm::Conv_Im2Col;
        if (str == "Conv_Winograd")
          return ComputeAlgorithm::Conv_Winograd;
        // Default case, could also throw an error
        return ComputeAlgorithm::Undefine;
      }

      StringRef getDataflowStrategyStrRef(DataflowStrategy strategy)
      {
        switch (strategy)
        {
        case DataflowStrategy::WeightStationary:
          return "WeightStationary";
          break;
        case DataflowStrategy::InputStationary:
          return "InputStationary";
          break;
        case DataflowStrategy::OutputStationary:
          return "OutputStationary";
          break;
        default:
          return "Undefine";
        }
      }

      StringRef getComputeAlgorithmStrRef(ComputeAlgorithm algorithm)
      {
        switch (algorithm)
        {
        case ComputeAlgorithm::GEMM_Standard:
          return "GEMM_Standard";
          break;
        case ComputeAlgorithm::Conv_Direct:
          return "Conv_Direct";
          break;
        case ComputeAlgorithm::Conv_Im2Col:
          return "Conv_Im2Col";
          break;
        case ComputeAlgorithm::Conv_Winograd:
          return "Conv_Winograd";
          break;
        default:
          return "Undefine";
        }
      }

      SystolicConfig parseSystolicConfig(Operation *op)
      {
        SystolicConfig config;

        // 1. Check the Interface implementation
        auto systolicOp = dyn_cast<ADORATensor::SystolicImplInterface>(op);
        assert(systolicOp && "Op must implement SystolicImplInterface");

        // 2. Parse Algorithm
        // Prefer Attribute; if empty, infer defaults from Op type
        StringRef algoStr = systolicOp.getAlgorithm();
        if (algoStr.empty())
        {
          if (isa<ADORATensor::GemmOp>(op) || isa<ADORATensor::MatMulOp>(op))
          {
            config.algorithm = ComputeAlgorithm::GEMM_Standard;
          }
          else if (isa<ADORATensor::ConvOp>(op))
          {
            // Conv defaults to Direct
            config.algorithm = ComputeAlgorithm::Conv_Direct;
          }
          else
          {
            config.algorithm = ComputeAlgorithm::Undefine;
          }
        }
        else
        {
          config.algorithm = parseAlgorithmStr(algoStr);
        }

        // 3. Parse Dataflow (Stationary Kind)
        StringRef dfStr = systolicOp.getStationaryKind();
        config.dataflow = parseDataflowStr(dfStr);
        // If undefined, default to WS for GEMM (adjust as needed)
        if (config.dataflow == DataflowStrategy::Undefine)
        {
          config.dataflow = DataflowStrategy::WeightStationary;
        }

        // 4. Parse TileSize
        auto tiles = systolicOp.getTileSize();
        config.tileSizes.assign(tiles.begin(), tiles.end());

        // 5. Parse LoopOrder
        auto order = systolicOp.getLoopOrder();
        if (!order.empty())
        {
          config.loopOrder.assign(order.begin(), order.end());
        }
        else
        {
          // Provide default LoopOrder
          if (config.algorithm == ComputeAlgorithm::Conv_Direct)
          {
            // Default N, K, P, Q, C, R, S -> 0,1,2,3,4,5,6
            config.loopOrder = {0, 1, 2, 3, 4, 5, 6};
          }
          else if (config.algorithm == ComputeAlgorithm::GEMM_Standard)
          {
            // Default M, N, K -> 0, 1, 2
            config.loopOrder = {0, 1, 2};
          }
        }

        return config;
      }

    } // end namespace ADORATensor
  } // end namespace ADORA

} // end namespace mlir
