//===------------------ GenericDirectConv.cpp - ADORATensor Lowering ------------------===//
/// builtin dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORA/Utility/Utility.h"

#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"

#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerConv.h"

using namespace ::mlir::ADORA::ADORATensor;
using namespace ::mlir::affine;
using namespace ::mlir;

namespace mlir
{
    namespace ADORA
    {
        namespace ADORATensor
        {

            // === 1. Implementation of Helpers ===

            // Extract metadata from ConvOp
            ConvMetadata getConvMetadata(ConvOp op)
            {
                ConvMetadata meta;

                // Get Shape: Input(N,C,H,W), Weight(K,C,R,S), Output(N,K,P,Q)
                auto inShape = op.getX().getType().cast<MemRefType>().getShape();
                auto wShape = op.getW().getType().cast<MemRefType>().getShape();
                auto outShape = op.getY().getType().cast<MemRefType>().getShape();

                // Fill Bounds [N, K, P, Q, C, R, S]
                meta.bounds.resize(7);
                meta.bounds[DimN] = outShape[0];
                meta.bounds[DimK] = outShape[1];
                meta.bounds[DimP] = outShape[2];
                meta.bounds[DimQ] = outShape[3];
                meta.bounds[DimC] = inShape[1];
                meta.bounds[DimR] = wShape[2];
                meta.bounds[DimS] = wShape[3];

                // Parse Strides
                if (auto s = op.getStrides())
                {
                    for (auto val : s.value())
                        meta.strides.push_back(val.cast<IntegerAttr>().getInt());
                }
                else
                {
                    meta.strides = {1, 1};
                }

                // Parse Dilations
                if (auto d = op.getDilations())
                {
                    for (auto val : d.value())
                        meta.dilations.push_back(val.cast<IntegerAttr>().getInt());
                }
                else
                {
                    meta.dilations = {1, 1};
                }

                // Pads (Simplified)
                meta.pads = {0, 0, 0, 0};

                meta.elementType = op.getX().getType().cast<MemRefType>().getElementType();
                return meta;
            }

            // === 2. Body Builder Implementation ===

            // Internal helper for loop body generation
            // Note: 'mutable' is added to lambda to fix const correctness issues with Op accessors
            static StationaryBodyBuilderFn TileofGenericDirectConv(
                ConvOp op,
                SystolicConfig config,
                const ConvMetadata &meta,
                const DenseMap<int, int> &loopIndexToDimMap // Map: LoopDepth -> ConvDim(0..6)
            )
            {
                return [=](OpBuilder &builder, Location loc, ValueRange ivs) mutable
                {
                    // ivs contains induction variables from outermost to innermost

                    // 1. Map IVs back to logical dimensions (N, K, P, Q...)
                    SmallVector<Value, 7> logicalIVs(7);
                    SmallVector<AffineExpr, 7> logicalExprs(7);

                    for (int i = 0; i < ivs.size(); ++i)
                    {
                        int logicalDim = loopIndexToDimMap.lookup(i);
                        logicalIVs[logicalDim] = ivs[i];
                        // Create affine expr for the i-th loop induction var
                        logicalExprs[logicalDim] = builder.getAffineDimExpr(i);
                    }

                    // 2. Build Access Map for Input (X)
                    // Formula: [n, c, p*stride_h + r*dilation_h, q*stride_w + s*dilation_w]
                    SmallVector<AffineExpr, 4> xExprs;
                    xExprs.push_back(logicalExprs[DimN]); // N
                    xExprs.push_back(logicalExprs[DimC]); // C

                    // H_in
                    AffineExpr hExpr = logicalExprs[DimP] * meta.strides[0] + logicalExprs[DimR] * meta.dilations[0];
                    xExprs.push_back(hExpr);

                    // W_in
                    AffineExpr wExpr = logicalExprs[DimQ] * meta.strides[1] + logicalExprs[DimS] * meta.dilations[1];
                    xExprs.push_back(wExpr);

                    AffineMap mapX = AffineMap::get(ivs.size(), 0, xExprs, builder.getContext());

                    // 3. Build Access Map for Weight (W)
                    // Formula: [k, c, r, s]
                    SmallVector<AffineExpr, 4> wExprs;
                    wExprs.push_back(logicalExprs[DimK]);
                    wExprs.push_back(logicalExprs[DimC]);
                    wExprs.push_back(logicalExprs[DimR]);
                    wExprs.push_back(logicalExprs[DimS]);
                    AffineMap mapW = AffineMap::get(ivs.size(), 0, wExprs, builder.getContext());

                    // 4. Build Access Map for Output (Y)
                    // Formula: [n, k, p, q]
                    SmallVector<AffineExpr, 4> yExprs;
                    yExprs.push_back(logicalExprs[DimN]);
                    yExprs.push_back(logicalExprs[DimK]);
                    yExprs.push_back(logicalExprs[DimP]);
                    yExprs.push_back(logicalExprs[DimQ]);
                    AffineMap mapY = AffineMap::get(ivs.size(), 0, yExprs, builder.getContext());

                    // === 3. Generate Computation Core (Affine Load/Store) ===

                    // Load X
                    auto loadX = builder.create<AffineLoadOp>(loc, op.getX(), mapX, ivs);
                    setPingpongAttr(loadX);

                    // Load W
                    auto loadW = builder.create<AffineLoadOp>(loc, op.getW(), mapW, ivs);
                    setPingpongAttr(loadW);

                    // Load Y (Accumulator)
                    auto loadY = builder.create<AffineLoadOp>(loc, op.getY(), mapY, ivs);
                    setPingpongAttr(loadY);

                    // Compute: Y += X * W
                    // Using helper functions from LowerGemm.h
                    Value mul = genArithMulOpAccordingToDataType(builder, loc, loadX, loadW)->getResult(0);
                    Value add = genArithAddOpAccordingToDataType(builder, loc, mul, loadY)->getResult(0);

                    // Store Y
                    auto storeY = builder.create<AffineStoreOp>(loc, add, op.getY(), mapY, ivs);
                    setPingpongAttr(storeY);
                };
            }

            // === 3. Main Entry Point ===

            mlir::affine::AffineForOp LowerGenericDirectConv(OpBuilder &b, ConvOp op, SystolicConfig config)
            {
                Location loc = op.getLoc();
                ConvMetadata meta = getConvMetadata(op);

                // 1. Prepare Loop Bounds and Steps
                // We need to reorder the logical bounds according to config.loopOrder

                SmallVector<int> sortedBounds;
                SmallVector<int> sortedSteps;

                // Key: Loop depth (0 is outermost), Value: ConvDim enum
                DenseMap<int, int> loopIndexToDimMap;

                if (config.loopOrder.size() != 7)
                {
                    llvm::errs() << "Error: loopOrder for Direct Conv must have 7 dimensions.\n";
                    return nullptr;
                }

                // Assume the last two dimensions in loopOrder map to spatial tiles (Row, Col)
                int spatialDimStart = config.loopOrder.size() - 2;

                for (int i = 0; i < config.loopOrder.size(); ++i)
                {
                    int logicalDim = config.loopOrder[i];

                    // Record mapping for BodyBuilder
                    loopIndexToDimMap[i] = logicalDim;

                    // Set Bounds
                    sortedBounds.push_back((int)meta.bounds[logicalDim]);

                    // Set Steps based on TileSize
                    if (i >= spatialDimStart)
                    {
                        int tileIdx = i - spatialDimStart;
                        if (tileIdx < config.tileSizes.size())
                        {
                            sortedSteps.push_back((int)config.tileSizes[tileIdx]);
                        }
                        else
                        {
                            sortedSteps.push_back(1);
                        }
                    }
                    else
                    {
                        sortedSteps.push_back(1);
                    }
                }

                // 2. Generate Loop Nest using shared utility from LowerGemm
                AffineForOp topLoop = OffDeviceNestedLoop(
                    b, loc,
                    /*level=*/(int)config.loopOrder.size(),
                    /*Upperbounds=*/sortedBounds,
                    /*Steps=*/sortedSteps,
                    /*InnerMostBodyBuilder=*/TileofGenericDirectConv(op, config, meta, loopIndexToDimMap));

                // 3. Post-processing
                SimplifyLoadStoreOpsInRegion(topLoop.getRegion());
                topLoop.walk([&](Operation *inst)
                             { inst->setAttr("ADORAConv", UnitAttr::get(topLoop.getContext())); });

                return topLoop;
            }

        } // namespace ADORATensor
    } // namespace ADORA
} // namespace mlir