//===------------------ mapGemm.cpp - ADORATensor Lower process ----------------------===//
/// builtin dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"

#include "tensorop/TensorOp.h"

using namespace ::mlir::ADORA::ADORATensor;
using namespace ::mlir::affine;

namespace mlir{
namespace ADORA{

static llvm::SmallVector<int64_t, 2> getShape(mlir::Value v) {
  mlir::Type type = v.getType();
  if (auto shapedTy = type.dyn_cast<mlir::ShapedType>()) {
    if (shapedTy.hasRank()) {
      return llvm::SmallVector<int64_t, 4>(shapedTy.getShape().begin(),
                                           shapedTy.getShape().end());
    }
  }
  return {};
}

using WSBodyBuilderFn = std::function<void(OpBuilder &, Location, ValueRange)>;

affine::AffineForOp GenerateOnDeviceNestedLoopWithoutLoopCarry(
    OpBuilder &builder, Location loc,
    int level = 2, SmallVector<int> Upperbounds = {1, 1}, 
    WSBodyBuilderFn InnerMostBodyBuilder = nullptr) 
{

  assert(level > 0 && "Loop nest level must be > 0");
  assert((int)Upperbounds.size() == level && 
         "Upperbounds size must == loop level");
  // create outermost loop 
  AffineForOp outer = builder.create<AffineForOp>(
      loc, /*lb*/ 0, /*ub*/ Upperbounds[0], /*step*/ 1);
  AffineForOp current = outer;

  SmallVector<Value> allIvs; /// collect itervar from every level
  allIvs.push_back(current.getInductionVar());

  // create inner loop one by one
  for (int i = 1; i < level; i++) {
    OpBuilder innerBuilder(current.getBody(),
                           std::prev(current.getBody()->end()));

    //// 
    if(i == level - 1){
      auto inner = innerBuilder.create<affine::AffineForOp>(
        // loc, 0, Upperbounds[i], 1, /*iterArgs =*/ ValueRange({}), InnerMostBodyBuilder);
        loc, (int64_t)0, (int64_t)Upperbounds[i], (int64_t)1, /*iterArgs =*/ ValueRange(), 
        [&](OpBuilder &b, Location loc, Value v, ValueRange vs) {
          /// v here is useless
          allIvs.push_back(v);
          assert(allIvs.size() >= 2);
          SmallVector<Value> lastTwoIvs(allIvs.end() - 2, allIvs.end());
          InnerMostBodyBuilder(b, loc, lastTwoIvs);
        });
      current = inner;
    }
    else{
      auto inner = innerBuilder.create<affine::AffineForOp>(
        loc, 0, Upperbounds[i], 1);
        
      current = inner;
      allIvs.push_back(current.getInductionVar());
    }
  }

  return outer;
}



/// @brief Loop body builder for the innermost tiled GEMM computation 
///        under Weight-Stationary (WS) dataflow. 
///
/// This function returns a body-builder lambda of type `WSBodyBuilderFn`, 
/// which is meant to be plugged into an `affine.for` nest. 
/// Within the loop body, affine load/store operations are generated to:
///   - Load an activation element from A, indexed by (i, k),
///   - Load a weight element from B, indexed by (k, j), 
///   - Load the current partial sum from C, indexed by (i, j),
///   - Perform multiply-accumulate (A * B + C),
///   - Store the updated result back into C.
///
/// The induction variables (`ivs`) are expected to come in the order {j, k, i}, 
/// corresponding to the outer loop nest over (N, K, M). 
/// For each loop iteration, the builder further expands a `tile_row_size × tile_col_size` 
/// micro-kernel of MAC operations inside the loop body.
///
/// @param A            MemRef value representing the activation/input tensor.
/// @param B            MemRef value representing the weight tensor (stationary).
/// @param C            MemRef value representing the output/accumulation tensor.
/// @param tile_row_size Number of rows in the tile (micro-kernel row dimension).
/// @param tile_col_size Number of columns in the tile (micro-kernel column dimension).
///
/// Example pseudocode for one tile iteration:
/// ```text
/// for j in N
///   for k in K
///     for i in M
///       for n in tile_row_size
///         for t in tile_col_size
///           C[i, j+n] += A[i, k+t] * B[k+t, j+n]
WSBodyBuilderFn InnermostBodyOfTiledWithWeightStationary(
    // OpBuilder builder,
    mlir::Value A,
    mlir::Value B,
    mlir::Value C,
    int tile_row_size,
    int tile_col_size
) {
  return [=](OpBuilder &builder, Location loc, ValueRange ivs) {
    // ivs = {j k i}
    assert(ivs.size() == 3 && "Expect 3 loop induction variables (i,j,k)");
    Value j_it = ivs[0];
    Value k_it = ivs[1];
    Value i_it = ivs[2];

    // j_it.dump();
    // k_it.dump();
    // i_it.dump();

    // auto i32Type = builder.getI32Type();

    /**
     * Example: N K M (j k i)
     * affine.for %arg3 = 0 to 9 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 36 {
            %0 = affine.load %arg0[%arg5, %arg4] : memref<?x36xi32>
            %1 = affine.load %arg1[%arg4, %arg3] : memref<?x36xi32>
            %2 = arith.muli %0, %1 : i32
            %3 = affine.load %arg2[%arg5, %arg3] : memref<?x36xi32>
            %4 = arith.addi %3, %2 : i32
            affine.store %4, %arg2[%arg5, %arg3] : memref<?x36xi32>
            ...
          }
        }
      }
     */
    for (int n = 0; n < tile_row_size; ++n) {
      for (int k = 0; k < tile_col_size; ++k) {
        /// affine map of A : 
        /// %0 = affine.load %arg0[%arg5, %arg4] : memref<?x36xi32>
        /// A[i, k]
        AffineExpr rowExpr_A = builder.getAffineDimExpr(0);
        AffineExpr colExpr_A = builder.getAffineDimExpr(1) + k;
        AffineMap map_A = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                  {rowExpr_A, colExpr_A}, builder.getContext());
        AffineLoadOp LoadA = builder.create<affine::AffineLoadOp>(
            loc, A, map_A, ValueRange{i_it, k_it});

        /// affine map of B : stationary
        /// %1 = affine.load %arg1[%arg4, %arg3] : memref<?x36xi32>
        /// B[k, j]
        AffineExpr rowExpr_B = builder.getAffineDimExpr(0) + k;
        AffineExpr colExpr_B = builder.getAffineDimExpr(1) + n;
        AffineMap map_B = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                  {rowExpr_B, colExpr_B}, builder.getContext());
        AffineLoadOp LoadB = builder.create<affine::AffineLoadOp>(
            loc, B, map_B, ValueRange{k_it, j_it});

        /// affine map of C
        /// %3 = affine.load %arg2[%arg5, %arg3] : memref<?x36xi32>
        /// C[i, j]
        AffineExpr rowExpr_C = builder.getAffineDimExpr(0);
        AffineExpr colExpr_C = builder.getAffineDimExpr(1) + n;
        AffineMap map_C = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                  {rowExpr_C, colExpr_C}, builder.getContext());
        AffineLoadOp LoadC = builder.create<affine::AffineLoadOp>(
            loc, C, map_C, ValueRange{i_it, j_it});

        //// get data type of mul and add
        Value mul, add;
        mlir::Type dtype = dyn_cast<::mlir::MemRefType>(A.getType()).getElementType();

        if(dtype.isSignlessInteger() || dtype.isSignedInteger() ||
          dtype.isUnsignedInteger()){
          mul = builder.create<arith::MulIOp>(loc, LoadA, LoadB);
          add = builder.create<arith::AddIOp>(loc, mul, LoadC);
        }
        else if(dtype.isBF16() || dtype.isF16() || dtype.isF32() || 
          dtype.isF64() || dtype.isF128()){
          mul = builder.create<arith::MulFOp>(loc, LoadA, LoadB);
          add = builder.create<arith::AddFOp>(loc, mul, LoadC);
        }
        else{
          assert(false && "Unsupprted data type.");
        }

        // store back to C
        AffineStoreOp StoreC = builder.create<affine::AffineStoreOp>(
            loc, add, C, map_C, ValueRange{i_it, j_it});
        
        LoadC.getOperation()->getBlock()->dump();
      }
    }

    builder.create<affine::AffineYieldOp>(loc);
  };
}



/// @brief Loop body builder for the innermost tiled GEMM computation 
///        under Weight-Stationary (WS) dataflow. 
///
/// This function returns a body-builder lambda of type `WSBodyBuilderFn`, 
/// which is meant to be plugged into an `affine.for` nest. 
/// Within the loop body, affine load/store operations are generated to:
///   - Load an activation element from A, indexed by (i, k),
///   - Load a weight element from B, indexed by (k, j), 
///   - Load the current partial sum from C, indexed by (i, j),
///   - Perform multiply-accumulate (A * B + C),
///   - Store the updated result back into C.
///
/// The induction variables (`ivs`) are expected to come in the order {j, k, i}, 
/// corresponding to the outer loop nest over (N, K, M). 
/// For each loop iteration, the builder further expands a `tile_row_size × tile_col_size` 
/// micro-kernel of MAC operations inside the loop body.
///
/// @param A            MemRef value representing the activation/input tensor.
/// @param B            MemRef value representing the weight tensor (stationary).
/// @param C            MemRef value representing the output/accumulation tensor.
/// @param tile_row_size Number of rows in the tile (micro-kernel row dimension).
/// @param tile_col_size Number of columns in the tile (micro-kernel column dimension).
///
/// Example pseudocode for one tile iteration:
/// ```text
/// for jj in N
///   for kk in K
///     //// followings are temporal map
///     for kkk in temporal_count_dim_k
///       (for ii in temporal_count_dim_m (might == M))
///       for i in temporal_count_dim_m
///         //// followings are spatial map
///         for n in tile_row_size
///           for t in tile_col_size
///             C[i, j+n] += A[i, k+t] * B[k+t, j+n]
WSBodyBuilderFn TileofWeightStationary(
    // OpBuilder builder,
    mlir::Value A,
    mlir::Value B,
    mlir::Value C,
    int temporal_count_dim_k,
    int temporal_count_dim_m,
    int tile_row_size,
    int tile_col_size
) {
  mlir::Type dtype = dyn_cast<::mlir::MemRefType>(A.getType()).getElementType();
  return [=](OpBuilder &builder, Location loc, ValueRange ivs) {
    /**
     * Example: N K [KK MM M]
     * affine.for %arg3 = 0 to 9 { /// N
        affine.for %arg4 = 0 to 9 {  /// K
          affine.for %arg5 = 0 to 4 {  /// M
            /// A matrix

        }
      }
     */
    mlir::Value vj = ivs[0];
    mlir::Value vk = ivs[1];
    mlir::Value vi = ivs[2];
    ///////////////////////////////////////////////
    ////// generate explicit data transfer: host to device
    ///////////////////////////////////////////////
    unsigned BlockLoadStoreOpId = 0;
    SmallVector<ADORA::DataBlockStoreOp> stores; 

    /////////////////////
    /// transfer A 
    /////////////////////
    for(int row = 0; row < tile_row_size; row++){
      SmallVector<AffineExpr, 2> Exprs;
      Exprs.push_back(builder.getAffineDimExpr(0) + row); // last dim's affine expr
      Exprs.push_back(builder.getAffineDimExpr(1)); // last dim's affine expr


      SmallVector<int64_t, 4> shape;
      shape.push_back(temporal_count_dim_k);
      shape.push_back(temporal_count_dim_m);

      AffineMap memIVmap = AffineMap::get(2, /*symbolCount=*/0, Exprs, builder.getContext());   /// stores corresponding AffineMap of above memIVs
      MemRefType newMemRef = MemRefType::get(shape, dtype);

      ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>\
                (loc, A, memIVmap, ValueRange({vi, vk}), newMemRef);
      // ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>\
                // (Kernel.getLoc(), memref, memIVmap, IVs, memRefType);
      // Kernel.getOperation()->getBlock()->push_back(BlockLoad);
      // BlockLoad.getOperation()->moveBefore(Kernel);
      BlockLoad.setKernelName("GEMMWS");
      BlockLoad.setId(std::to_string(BlockLoadStoreOpId++));

      /// has stride
      if(temporal_count_dim_m != 1){
        BlockLoad.setStrides(ArrayRef<int64_t>({1, tile_row_size}));
      }
    }

    /////////////////////
    /// transfer B 
    /////////////////////
    for(int row = 0; row < tile_row_size; row++){
      SmallVector<AffineExpr, 2> Exprs;
      Exprs.push_back(builder.getAffineDimExpr(0) + row); // last dim's affine expr
      
      if(tile_col_size >= 4){
        for(int col = 0; col < tile_col_size / 4; col++){
          SmallVector<AffineExpr, 2> Exprs;
          Exprs.push_back(builder.getAffineDimExpr(0) + row); // last dim's affine expr
          Exprs.push_back(builder.getAffineDimExpr(1) + col*4); // last dim's affine expr

          SmallVector<int64_t, 4> shape;
          shape.push_back(temporal_count_dim_k);
          shape.push_back(4);

          AffineMap memIVmap = AffineMap::get(2, /*symbolCount=*/0, Exprs, builder.getContext());   /// stores corresponding AffineMap of above memIVs
          MemRefType newMemRef = MemRefType::get(shape, dtype);
          // llvm::errs() << "[debug] newMemRef: ";newMemRef.dump();

          ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>\
                    (loc, A, memIVmap, ValueRange({vk, vj}), newMemRef);
          // ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>\
                    // (Kernel.getLoc(), memref, memIVmap, IVs, memRefType);
          // Kernel.getOperation()->getBlock()->push_back(BlockLoad);
          // BlockLoad.getOperation()->moveBefore(Kernel);
          BlockLoad.setKernelName("GEMMWS");
          BlockLoad.setId(std::to_string(BlockLoadStoreOpId++));

          /// has stride
          if(temporal_count_dim_k != 1){
            BlockLoad.setStrides(ArrayRef<int64_t>({tile_row_size, 1}));
          }
        }
      }

      //// the remaining col%4 or col < 4
      {
        SmallVector<AffineExpr, 2> Exprs;
        Exprs.push_back(builder.getAffineDimExpr(0) + row); // last dim's affine expr
        Exprs.push_back(builder.getAffineDimExpr(1) + (int)tile_col_size/(int)4); // last dim's affine expr

        SmallVector<int64_t, 4> shape;
        shape.push_back(temporal_count_dim_k);
        shape.push_back(tile_col_size%4);

        AffineMap memIVmap = AffineMap::get(2, /*symbolCount=*/0, Exprs, builder.getContext());   /// stores corresponding AffineMap of above memIVs
        MemRefType newMemRef = MemRefType::get(shape, dtype);

        ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>\
                  (loc, B, memIVmap, ValueRange({vk, vj}), newMemRef);
        // ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>\
                  // (Kernel.getLoc(), memref, memIVmap, IVs, memRefType);
        // Kernel.getOperation()->getBlock()->push_back(BlockLoad);
        // BlockLoad.getOperation()->moveBefore(Kernel);
        BlockLoad.setKernelName("GEMMWS");
        BlockLoad.setId(std::to_string(BlockLoadStoreOpId++));

        /// has stride
        if(temporal_count_dim_k != 1){
          BlockLoad.setStrides(ArrayRef<int64_t>({tile_row_size, 1}));
        }
      }
    }
    /////////////////////
    /// End of transfer B 
    /////////////////////

    /////////////////////
    /// transfer C
    /////////////////////
    for(int col = 0; col < tile_col_size; col++){
      SmallVector<AffineExpr, 2> Exprs;
      Exprs.push_back(builder.getAffineDimExpr(0) + col); // last dim's affine expr
      Exprs.push_back(builder.getAffineDimExpr(1)); // last dim's affine expr

      SmallVector<int64_t, 4> shape;
      shape.push_back(temporal_count_dim_m);
      shape.push_back(1);

      AffineMap memIVmap = AffineMap::get(2, /*symbolCount=*/0, Exprs, builder.getContext());   /// stores corresponding AffineMap of above memIVs
      MemRefType newMemRef = MemRefType::get(shape, dtype);

      ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>\
              (loc, C, memIVmap, ValueRange({vi, vj}), newMemRef);
      // ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>\
                    // (Kernel.getLoc(), memref, memIVmap, IVs, memRefType);
      // Kernel.getOperation()->getBlock()->push_back(BlockLoad);
      // BlockLoad.getOperation()->moveBefore(Kernel);
      
      BlockLoad.setKernelName("GEMMWS");
      BlockLoad.setId(std::to_string(BlockLoadStoreOpId++));

      /// has stride
      // if(temporal_count_dim_k != 1){
      //   BlockLoad.setStrides(ArrayRef<int64_t>({tile_row_size, 1}));
      // }
      ///////////
      /// generate the store back
      ///////////
      ADORA::LocalMemAllocOp alloc = builder.create<ADORA::LocalMemAllocOp>(newMemRef);
      alloc.setKernelName(Kernel.getKernelName());
      alloc.setId(std::to_string(BlockLoadStoreOpId));


      ADORA::DataBlockStoreOp BlockStore = builder.create<ADORA::DataBlockStoreOp>\
              (loc, BlockLoad, C, memIVmap, ValueRange({vi, vj}));
      // ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>\
                    // (Kernel.getLoc(), memref, memIVmap, IVs, memRefType);
      // Kernel.getOperation()->getBlock()->push_back(BlockLoad);
      // BlockLoad.getOperation()->moveBefore(Kernel);
      
      BlockStore.setKernelName("GEMMWS");
      BlockStore.setId(std::to_string(BlockLoadStoreOpId++));     

      stores.push_back(BlockStore); 
    }   
    /////////////////////
    /// End of transfer C
    /////////////////////

    /////////////////////
    /// start generate inner loop
    /////////////////////
    AffineForOp loop = GenerateOnDeviceNestedLoopWithoutLoopCarry(
      builder, loc, 
      /*level*/2, 
      /*upper bounds*/{temporal_count_dim_k, temporal_count_dim_m}, //// K -> M
      /*InnerMostBodyBuilder*/InnermostBodyOfTiledWithWeightStationary(
        A, B, C, tile_row_size, tile_col_size
      )
    );

    affine::AffineYieldOp yield = builder.create<affine::AffineYieldOp>(loc);


    for(auto store : stores){
      store.getOperation()->moveBefore(yield);
    }


  };

}

/// @brief Generate nested loop which is the outer loops out of a systolic tile
///
/// @param A            MemRef value representing the activation/input tensor.
/// @param B            MemRef value representing the weight tensor (stationary).
/// @param C            MemRef value representing the output/accumulation tensor.
/// @param tile_row_size Number of rows in the tile (micro-kernel row dimension).
/// @param tile_col_size Number of columns in the tile (micro-kernel column dimension).
affine::AffineForOp OffDeviceLoopOfWeightStationary(
    OpBuilder &builder, Location loc,
    int level = 1, SmallVector<int> Upperbounds = {1}, 
    WSBodyBuilderFn InnerMostBodyBuilder = nullptr) 
{

  assert(level > 0 && "Loop nest level must be > 0");
  assert((int)Upperbounds.size() == level && 
         "Upperbounds size must == loop level");

  // create outermost loop 
  AffineForOp outer = builder.create<AffineForOp>(
      loc, /*lb*/ 0, /*ub*/ Upperbounds[0], /*step*/ 1);
  AffineForOp current = outer;

  SmallVector<Value> allIvs; /// collect itervar from every level
  allIvs.push_back(current.getInductionVar());

  // create inner loop one by one
  for (int i = 1; i < level; i++) {
    OpBuilder innerBuilder(current.getBody(),
                           std::prev(current.getBody()->end()));

    //// 
    if(i == level - 1){
      auto inner = innerBuilder.create<affine::AffineForOp>(
        // loc, 0, Upperbounds[i], 1, /*iterArgs =*/ ValueRange({}), InnerMostBodyBuilder);
        loc, (int64_t)0, (int64_t)Upperbounds[i], (int64_t)1, /*iterArgs =*/ ValueRange(), 
        [&](OpBuilder &b, Location loc, Value v, ValueRange vs) {
          /// v here is useless
          allIvs.push_back(v);
          assert(allIvs.size() >= 3);
          SmallVector<Value> lastThreeIvs(allIvs.end() - 3, allIvs.end());
          InnerMostBodyBuilder(b, loc, lastThreeIvs);
        });
      current = inner;
    }
    else{
      auto inner = innerBuilder.create<affine::AffineForOp>(
        loc, 0, Upperbounds[i], 1);
        
      current = inner;
      allIvs.push_back(current.getInductionVar());
    }
  }

  return outer;
}


/////
//// WS Order : N (j) -> K (k) -> M (i)
////  N - col of B, col of C
////  K - reduction dim
////  M - row of A, row of C
//// when tile size = 4, (K_temporal_tile, M_temporal_tile, K_spatial_tile, N_spatial_tile)
//// when tile size = 3, (M_temporal_tile, K_spatial_tile, N_spatial_tile), K_temporal_tile == 1
//// when tile size = 2, (K_spatial_tile, N_spatial_tile), K_temporal_tile == 1, M_temporal_tile = M
AffineForOp TiledWeightStationaryGemm(
  OpBuilder opbuilder,
  ADORATensor::GemmOp op, 
  ArrayRef<int64_t> tilesize //(K_temporal_tile, M_temporal_tile, K_spatial_tile, N_spatial_tile)
){
  assert(tilesize.size() == 2 || tilesize.size() == 3 ||  tilesize.size() == 4);

  llvm::SmallVector<int64_t, 2> ShapeA = getShape(op.getA());
  llvm::SmallVector<int64_t, 2> ShapeB = getShape(op.getB());
  llvm::SmallVector<int64_t, 2> ShapeC = getShape(op.getC());

  //////////////////////////////////////
  /// Get tiled matmul micro-kernel parameter, which is also the tile of B matrix 
  //////////////////////////////////////
  int64_t tilerow, tilecol, K_temporal_tile, M_temporal_tile;
  tilerow = tilesize[tilesize.size() - 2]; /// the tile of micro kernel row, also the tile of B's row
  tilecol = tilesize[tilesize.size() - 1]; /// the tile of micro kernel col, also the tile of B's col

  if(tilesize.size() == 3){
    K_temporal_tile = 1;
    M_temporal_tile = tilesize[0];
  }
  else if(tilesize.size() == 4){
    K_temporal_tile = tilesize[0];
    M_temporal_tile = tilesize[1];
  }
  else{
    K_temporal_tile = 1;
    M_temporal_tile = ShapeA[0];    
  }

  // make sure matmul is legal
  assert(ShapeA.size() == 2 && ShapeB.size() == 2 && ShapeC.size() == 2);
  assert(ShapeA[0] == ShapeC[0] && ShapeA[1] == ShapeB[0] && ShapeB[1] == ShapeC[1]);
  assert(ShapeA[1] % (tilerow * K_temporal_tile) == 0 
      && ShapeB[1] % tilecol == 0 
      && ShapeA[0] % M_temporal_tile == 0);
  
  if(ShapeA[0] != M_temporal_tile){
    assert(K_temporal_tile == 1);
  }
  
  int64_t M_aftertile, N_aftertile, K_aftertile;

  M_aftertile = ShapeA[0] / M_temporal_tile;
  N_aftertile = ShapeB[1] / tilecol;
  K_aftertile = ShapeB[0] / (tilecol * K_temporal_tile);

  //////////////////////////////////////
  /// Generate systolic gemm
  //////////////////////////////////////
  AffineForOp loop;
    loop = OffDeviceLoopOfWeightStationary(
      opbuilder, op.getLoc(), 
      /*level*/3, 
      /*upper bounds*/{N_aftertile, K_aftertile, M_aftertile}, //// N -> K -> M
      /*InnerMostBodyBuilder*/TileofWeightStationary(
        op.getA(), op.getB(), op.getC(), K_temporal_tile, M_temporal_tile, tilerow, tilecol
      )
    );
  
  op.getOperation()->getBlock()->push_back(loop);
  loop.getOperation()->moveAfter(op);
  // } 
  // else{
  //   loop = GenerateTiledNestedLoopWithoutLoopCarry(
  //     builder, op.getLoc(), 
  //     /*level*/4, 
  //     /*upper bounds*/{N_aftertile, K_aftertile, M_aftertile, tiletemporal}, //// M -> N -> K -> MM
  //     /*InnerMostBodyBuilder*/InnermostBodyOfTiledWithWeightStationary(
  //       op.getA(), op.getB(), op.getC(), tilerow, tilecol
  //     )
  //   );    
  // }


  loop.dump();

  return loop;
}

}
}