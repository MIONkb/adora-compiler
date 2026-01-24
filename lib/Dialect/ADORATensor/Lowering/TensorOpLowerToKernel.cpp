//===----------------------------------------------------------------------===//
//
// This file implements Data flow graph generation for ADORA Tensor
//
//===----------------------------------------------------------------------===//

/// builtin dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Misc/DFG.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOpLowerToKernel.h"
// #include "PassDetail.h"

// lower tensor op
#include "../../../../mapper/include/tensorop/TensorOp.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerGemm.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerConv.h"
// // DFG
// ADORA::KernelOp* _kernel_toDFG;
// int _variable_config_cnt = 0;

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;

#define DEBUG_TYPE "adora-tensor-op-lower"

namespace mlir
{
  namespace ADORA
  {
    namespace ADORATensor
    {

      bool TensorOpCDFGVisitor::visitOp(ADORATensor::GemmOp op)
      {
        SystolicImplInterface SystolicPara(op);
        ArrayRef<int64_t> tilesize = SystolicPara.getTileSize();
        StringRef strategy = SystolicPara.getStationaryKind();
        AffineForOp newfor;
        if (strategy == getDataflowStrategyStrRef(DataflowStrategy::WeightStationary))
        {
          newfor = TiledWeightStationaryGemm(opbuilder, op, tilesize);
        }
        else if (strategy == getDataflowStrategyStrRef(DataflowStrategy::InputStationary))
        {
          newfor = TiledInputStationaryGemm(opbuilder, op, tilesize);
        }
        else if (strategy == getDataflowStrategyStrRef(DataflowStrategy::OutputStationary))
        {
          newfor = TiledOutputStationaryGemm(opbuilder, op, tilesize);
        }

        return true;
      }

      bool TensorOpCDFGVisitor::visitOp(ADORATensor::ConvOp op)
      {
        // 1. 统一解析配置
        // 这一步屏蔽了复杂的 Attribute 读取细节
        SystolicConfig config = parseSystolicConfig(op);

        AffineForOp newfor;

        // 2. 根据算法一级分发
        switch (config.algorithm)
        {
        case ComputeAlgorithm::Conv_Direct:
          // 调用 Direct Conv 生成器
          // 这里的 LowerGenericDirectConv 会读取 config.loopOrder
          // 来决定是生成 OS (P,Q在外) 还是 WS (R,S在外)
          newfor = LowerGenericDirectConv(opbuilder, op, config);
          break;

        case ComputeAlgorithm::Conv_Im2Col:
          // Im2Col 通常涉及先 Transform 再 GEMM
          // newfor = LowerIm2ColConv(opbuilder, op, config);
          break;

        case ComputeAlgorithm::Conv_Winograd:
          // Winograd 变换 -> 矩阵乘 -> 逆变换
          // newfor = LowerWinogradConv(opbuilder, op, config);
          break;

        case ComputeAlgorithm::GEMM_Standard:
          llvm::errs() << "[Error] ConvOp should not use GEMM_Standard algorithm directly.\n";
          return false;

        default:
          llvm::errs() << "[Error] Unknown Conv Algorithm.\n";
          return false;
        }

        return true;
      }

    }
  }
}