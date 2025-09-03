//===------------------ NestedLoopGeneration.cpp - ADORATensor Lower process util functions ----------------------===//
/// builtin dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORA/Utility/Utility.h"

#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"

#include "tensorop/TensorOp.h"
#include "tensorop/MapGemm.h"

using namespace ::mlir::ADORA::ADORATensor;
using namespace ::mlir::affine;

namespace mlir{
namespace ADORA{

mlir::Value getConstantOpAccordingToDataType(OpBuilder &builder, Location loc, Type datatype, float value) {
  arith::ConstantOp newconst;
  if(datatype.isBF16()||datatype.isF32()) {
    float constvalue = value;
    FloatAttr constAttr = FloatAttr::get(datatype, constvalue);
    newconst = builder.create<arith::ConstantOp>(loc, datatype , constAttr);
  }
  else if(datatype.isF64()){
    double constvalue = (double)value;
    FloatAttr constAttr = FloatAttr::get(datatype, constvalue);
    newconst = builder.create<arith::ConstantOp>(loc, datatype , constAttr);
  }
  else {
    int64_t constvalue = (int)value;
    IntegerAttr constAttr = IntegerAttr::get(datatype, constvalue);
    newconst = builder.create<arith::ConstantOp>(loc, datatype , constAttr);    
  }
  return newconst.getResult();
}

/// @brief Create an `arith.add` op for the given operands.  
/// 
/// Selects `AddFOp` for floating-point types (bf16/f32/f64),  
/// otherwise uses `AddIOp` for integer types.  
/// Returns the created operation.
mlir::Operation* genArithAddOpAccordingToDataType(OpBuilder &builder, Location loc, mlir::Value lhs, mlir::Value rhs) {
  assert(lhs.getType() == rhs.getType());
  mlir::Type datatype;
  if(isa<mlir::VectorType>(lhs.getType())){
    datatype = dyn_cast <mlir::VectorType> (lhs.getType()).getElementType();
  }
  else{
    datatype = lhs.getType();
  }

  mlir::Operation* add;
  if(datatype.isBF16()|| datatype.isF32() || datatype.isF64()) {
    add = builder.create<arith::AddFOp>(loc, lhs, rhs);
  }
  else {
    add = builder.create<arith::AddIOp>(loc, lhs, rhs);
  }
  return add;
}

mlir::Operation* genArithMulOpAccordingToDataType(OpBuilder &builder, Location loc, mlir::Value lhs, mlir::Value rhs) {
  assert(lhs.getType() == rhs.getType());
  mlir::Type datatype;
  if(isa<mlir::VectorType>(lhs.getType())){
    datatype = dyn_cast <mlir::VectorType> (lhs.getType()).getElementType();
  }
  else{
    datatype = lhs.getType();
  }
  
  mlir::Operation* mul;
  if(datatype.isBF16()|| datatype.isF32() || datatype.isF64()) {
    mul = builder.create<arith::MulFOp>(loc, lhs, rhs);
  }
  else {
    mul = builder.create<arith::MulIOp>(loc, lhs, rhs);
  }
  return mul;
}



/// @brief Generate a nested affine.for loop on device(data transfer is already done).
///  This function is shared by IS and WS
/// 
/// Creates a loop nest of depth `level` with upper bounds
/// specified by `Upperbounds`, and invokes the provided
/// `BodyBuilder` at the innermost level.  
/// The function returns the outermost `AffineForOp`.
///
/// @param builder     MLIR OpBuilder for loop creation
/// @param loc         Source location
/// @param level       Nesting depth (must > 0)
/// @param Upperbounds Upper bounds per loop level
/// @param BodyBuilder Callback to build loop body
affine::AffineForOp GenerateOnDeviceNestedLoop(
    OpBuilder &builder, Location loc,
    int level, SmallVector<int> Upperbounds, 
    StationaryBodyBuilderFn BodyBuilder) 
{

  assert(level > 0 && "Loop nest level must be > 0");
  assert((int)Upperbounds.size() == level && 
         "Upperbounds size must == loop level");
  // create outermost loop 
  SmallVector<Value> allIvs;
  AffineForOp outer = builder.create<AffineForOp>(
    loc, /*lb*/ 0, /*ub*/ Upperbounds[0], /*step*/ 1, /*iterArgs =*/ ValueRange(), 
      [&](OpBuilder &b, Location loc, Value v, ValueRange vs) {
        /// v here is useless
        allIvs.push_back(v);
        // assert(allIvs.size() >= 2);
        SmallVector<Value> OutermostIvs(allIvs.end() - 1, allIvs.end());
        BodyBuilder(b, loc, OutermostIvs);
    });

  return outer;
}


/// @brief Generate nested loop which is the outer loops out of a systolic tile
///  Shared by three stationary dataflow.
/// @param A            MemRef value representing the activation/input tensor.
/// @param B            MemRef value representing the weight tensor (stationary).
/// @param C            MemRef value representing the output/accumulation tensor.
/// @param tile_row_size Number of rows in the tile (micro-kernel row dimension).
/// @param tile_col_size Number of columns in the tile (micro-kernel column dimension).
affine::AffineForOp OffDeviceNestedLoop(
    OpBuilder &builder, Location loc,
    int level, 
    SmallVector<int> Upperbounds, 
    SmallVector<int> Steps, 
    StationaryBodyBuilderFn InnerMostBodyBuilder) 
{

  assert(level > 0 && "Loop nest level must be > 0");
  assert((int)Upperbounds.size() == level && "Upperbounds size must == loop level");
  assert((int)Steps.size() == level && "Steps size must == loop level");

  // create outermost loop 
  AffineForOp outer = builder.create<AffineForOp>(
      loc, /*lb*/ 0, /*ub*/ Upperbounds[0], /*step*/ Steps[0]);
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
        loc, (int64_t)0, (int64_t)Upperbounds[i], /*step*/(int64_t)Steps[i], /*iterArgs =*/ ValueRange(), 
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
        loc, 0, Upperbounds[i], /*step*/(int64_t)Steps[i]);
        
      current = inner;
      allIvs.push_back(current.getInductionVar());
    }
  }

  return outer;
}

}
}