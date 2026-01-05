//===----------------------------------------------------------------------===//
//
// This file implements automatic dataflow strategy decision for ADORA Tensor GemmOp
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/OperationSupport.h"
// #include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Support/LLVM.h"

#include <iostream>
#include <string>
#include <bit>
#include <math.h>

// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

// For op transformation
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "PassDetail.h"
// lower tensor op
// #include "../../../../mapper/include/tensorop/TensorOp.h"


using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;

#define DEBUG_TYPE "adora-tensor-gemm-strategy-decision"

namespace mlir{
namespace ADORA{
namespace ADORATensor{

class ADORAGemmOpStrategyDecisionPass : public ADORAGemmOpStrategyDecisionPassBase<ADORAGemmOpStrategyDecisionPass>
{
  void runOnOperation() override;
};

std::unique_ptr<OperationPass<ModuleOp>> createADORAGemmOpStrategyDecisionPass()
{
  return std::make_unique<ADORAGemmOpStrategyDecisionPass>();
}

// ceiling division
inline int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// element bytes from MLIR type (int/float); fallback 4B
int64_t elemBytes(mlir::Type t) {
  if (auto it = t.dyn_cast<mlir::IntegerType>()) return it.getWidth() / 8;
  if (auto ft = t.dyn_cast<mlir::FloatType>())   return ft.getWidth() / 8;
  return 4;
}

std::vector<std::pair<int,int>> findFeasibleSpatialMap(
  int num_pe, int num_iob, int limit = 80
){
  std::vector<std::pair<int,int>> cand;
  for (int r = 1; r <= num_iob; ++r) {
    for(int c = 1; c <= num_iob; ++c){
      if(r*c < num_pe && 
        (r + 2*c <= num_iob || 2*r + c <= num_iob))
      {
        cand.emplace_back(r, c);
      }
    }
  }

  // also include a few "near-factor" pairs by shaving columns
  // int num_pe = std::max(1, num_pe / std::max(1, (int)std::sqrt(num_pe)));
  // for (int r = 1; r <= num_iob; ++r) {
  //   int c = num_pe / r;
  //   cand.emplace_back(r, c);
  // }
  // uniq + clamp size
  std::sort(cand.begin(), cand.end());
  cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
  std::sort(cand.begin(), cand.end(), [](auto a, auto b){
    double sa = a.first * a.second;
    double sb = a.first * a.second;
    return sa < sb;
  });
  if ((int)cand.size() > limit) cand.resize(limit);
  return cand;

}

bool legalRowCol(MatMulStrategy stationarykind, std::pair <int, int> sa, int num_pe, int num_io){
  /////// check PE limits and IO 
  int row = sa.first;
  int col = sa.second;
  int io_cost;
  bool legal = false;
  switch (stationarykind)
  {
  case MatMulStrategy::InputStationary :
    io_cost = row*col / 4 + col + 2 * row;
    if(row * (2 + col) <= num_pe 
      && io_cost <= num_io){
      legal = true;
    }
    break;
  case MatMulStrategy::WeightStationary :
    io_cost = row*col / 4 + col * 2 + row;
    if(col * (2 + row) <= num_pe 
      && io_cost <= num_io){
      legal = true;
    }
    break;
  case MatMulStrategy::OutputStationary :
    io_cost = 2 * row*col / 4  + col + row;
    if(col * (2 + row) <= num_pe 
      && io_cost <= num_io){
      legal = true;
    }
    break;  
  default:
    assert(false && "No defined MatMulStrategy");
    break;
  }
  return legal;
}

// largest divisor of x that is <= bound (returns >=1)
int64_t largestDivisorLE(int64_t x, int64_t bound) {
  bound = std::max<int64_t>(1, std::min(x, bound));
  for (int64_t d = bound; d >= 1; --d) if (x % d == 0) return d;
  return 1;
}

  // // simple utilization score: how well (row,col) fits (Rmapped,Cmapped)
  // double utilScore(int64_t row, int64_t col, int64_t Rmap, int64_t Cmap) {
  //   double ur = std::min<double>(1.0, (double)Rmap / std::max<int64_t>(1,row));
  //   double uc = std::min<double>(1.0, (double)Cmap / std::max<int64_t>(1,col));
  //   return 0.5*ur + 0.5*uc;
  // }

// choose which dims map to (row,col) under each strategy (heuristic)
struct MapDims { int64_t Rmap, Cmap, innermostNonStationary, T0_dimension; };
MapDims mappingFor(MatMulStrategy s, int64_t M, int64_t N, int64_t K) {
  switch (s) {
    case MatMulStrategy::InputStationary: // A is MxK; stream along N; K reduced inside
      return { /*Rmap=*/M, /*Cmap=*/K, /*innermostNonStationary=*/N, /*T0_dimension=*/M };
    case MatMulStrategy::WeightStationary: // B is KxN; stream along M
      return { /*Rmap=*/K, /*Cmap=*/N, /*innermostNonStationary=*/M, /*T0_dimension=*/K };
    case MatMulStrategy::OutputStationary: // keep C; reduce K while streaming both A/B
      return { /*Rmap=*/M, /*Cmap=*/N, /*innermostNonStationary=*/K, /*T0_dimension=*/N };
    default:
      return { M, N, K };
  }
}

/// @brief A utility function to check if a block access operation can be simplified
/// @param op The candidate operation, either a datablockload or datablockstore
/// @return True if the operation is simplified, otherwise false
int64_t getByteSizeFromMemref(mlir::MemRefType memref) {
    mlir::Type elementType = memref.getElementType();

    if (!elementType.isIntOrFloat()) {
        llvm::errs() << "Unsupported element type in MemRef: " << elementType << "\n";
        return -1;
    }
    unsigned elemBitWidth = elementType.getIntOrFloatBitWidth();

    int64_t numElements = 1;
    for (int64_t dim : memref.getShape()) {
        if (dim == mlir::ShapedType::kDynamic) {
            llvm::errs() << "Dynamic dimension found in MemRef: " << memref << "\n";
            assert(false);
        }
        numElements *= dim;
    }

    int64_t totalBytes = (elemBitWidth / 8) * numElements;
    return totalBytes;
}

int64_t getByteSizeFromMemref(mlir::TypedValue<mlir::MemRefType> memref) {
  return getByteSizeFromMemref(memref.getType());
}

static int64_t getDataBytes(mlir::Type t){
  if(isa<mlir::MemRefType>(t)){
    return t.cast<mlir::MemRefType>().getElementTypeBitWidth()/8;
  }
  else if(isa<mlir::RankedTensorType>(t)){
    return t.cast<mlir::RankedTensorType>().getElementTypeBitWidth()/8;
  }
  else{
    assert(false && "Unsupported type.");
  }
}

static ArrayRef<int64_t> getShape(mlir::Type t){
  if(isa<mlir::MemRefType>(t)){
    return t.cast<mlir::MemRefType>().getShape();
  }
  else if(isa<mlir::RankedTensorType>(t)){
    return t.cast<mlir::RankedTensorType>().getShape();
  }
  else{
    assert(false && "Unsupported type.");
  }
}

/// Trim a shape to 2D if possible.
/// - If shape is [1, M, N] (or [0, M, N]), drop the leading dimension
/// - If shape is already 2D, keep it
/// - Otherwise, return the original shape (or assert, depending on policy)
static inline llvm::SmallVector<int64_t, 2>
trimTo2D(llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t, 2> result;

  if (shape.size() == 3 && (shape[0] == 1 || shape[0] == 0)) {
    // [1, M, N] -> [M, N]
    result.push_back(shape[1]);
    result.push_back(shape[2]);
    return result;
  }

  else if (shape.size() == 2) {
    // [M, N] -> [M, N]
    result.push_back(shape[0]);
    result.push_back(shape[1]);
    return result;
  }

  else if (shape.size() == 1) {
    result.push_back(shape[0]);
    return result;
  }

  // Fallback policy (choose ONE):
  // 1) Be permissive: take the last two dims
  if (shape.size() > 2) {
    result.push_back(shape[shape.size() - 2]);
    result.push_back(shape[shape.size() - 1]);
    return result;
  }

  // 2) Or be strict:
  // assert(false && "trimTo2D: unsupported shape rank");

  return result;
}

// pipeline fill estimate for a rowxcol systolic array
inline int64_t pipeFill(int64_t row, int64_t col) {

  return std::max<int64_t>(0, (int)lround(2.0 * sqrt((double)row * col))); 
}

// EC cycles with reconfig amortization
int64_t execCycles(MatMulStrategy s, int64_t M, int64_t N, int64_t K,
                     int64_t row, int64_t col, int64_t T0, int64_t T1,
                     int numPEs, int numIOBs) {

  int64_t mac = (M * N * K) / std::max<int64_t>(1, row * col);
  int cgra_col = numIOBs/2;
  int cgra_row = numPEs/cgra_col;
  int64_t S = pipeFill(cgra_row, cgra_col);

  long double reconfig = 3*row + col + S; // follow paper's IS derivation style
  switch (s) {
    case MatMulStrategy::InputStationary: {
      reconfig = (long double)(3*row + col + S);
      break;
    }
    case MatMulStrategy::WeightStationary:{
      reconfig = (long double)(3*col + row + S);
      break;
    }
    case MatMulStrategy::OutputStationary:{
      reconfig = (long double)(3*col + row + S);
      break;
    }
    default: {
      reconfig = (long double)(3*row + col + S);
      break;      
    }
  }
  // amortize by T0*T1
  return mac * (1 + (reconfig)/((long double)(T0 * T1)));
}

// Transfer volume (bytes). IS uses your analytic; others fallback to one-pass baseline.
int64_t transferVolumn(MatMulStrategy s, int64_t M, int64_t N, int64_t K,
                        int64_t row, int64_t col) {
  switch (s) {
    case MatMulStrategy::InputStationary: {
      long double tv_elems = (long double)M*K*N * (1.0/N + 1.0/M + 1.0/col);
      return (int64_t)tv_elems;
    }
    case MatMulStrategy::WeightStationary:{
      long double tv_elems = (long double)M*K*N * (1.0/col + 1.0/M + 1.0/K);
      return (int64_t)tv_elems;
    }
    case MatMulStrategy::OutputStationary:{
      long double tv_elems = (long double)M*K*N * (1.0/N + 1.0/row + 1.0/K);
      return (int64_t)tv_elems;
    }
    default: {
      // Baseline: one-time host-device transfers of A,B,C
      // long double tv_elems = (long double)M*K + (long double)K*N + (long double)M*N;
      // weight by respective element sizes
      return M*K + K*N + M*N;
    }
  }
}

bool CheckTileDivided(MatMulStrategy s, int64_t M, int64_t N, int64_t K,
                        int64_t row, int64_t col, int64_t T0, int64_t T1) {
  switch (s) {
    case MatMulStrategy::InputStationary: {
      bool divided = M%(T0*row) == 0 && N%(T1) == 0 && K%(col) == 0;
      return divided;
    }
    case MatMulStrategy::WeightStationary:{
      bool divided = M%(T1) == 0 && N%(col) == 0 && K%(T0*row) == 0;
      return divided;
    }
    case MatMulStrategy::OutputStationary:{
      bool divided = M%(row) == 0 && N%(T0*col) == 0 && K%(T1) == 0;
      return divided;
    }
    default: {
      // Baseline: one-time host-device transfers of A,B,C
      // long double tv_elems = (long double)M*K + (long double)K*N + (long double)M*N;
      // weight by respective element sizes
      abort();
    }
  }
}

static inline llvm::SmallVector<int64_t, 32>
getAllDivisor(int64_t n) {
  llvm::SmallVector<int64_t, 32> divs;
  if (n == 0) {          
    divs.push_back(1); 
    return divs;
  }
  if (n < 0) n = -n;     

  int64_t r = static_cast<int64_t>(std::sqrt(static_cast<long double>(n)));
  for (int64_t d = 1; d <= r; ++d) {
    if (n % d == 0) {
      divs.push_back(d);
      int64_t other = n / d;
      if (other != d) divs.push_back(other);
    }
  }

  llvm::sort(divs.begin(), divs.end());
  divs.erase(std::unique(divs.begin(), divs.end()), divs.end());
  return divs;
}

void AutoSetDataflowStrategy(
    mlir::ADORA::ADORATensor::GemmOp gemmop, 
    int num_pe, int num_iob, int bank_byte, int BandByteWidth, 
    MatMulStrategy _strategy = MatMulStrategy::Undefine
  ){
  // Get the input tensors
  auto A = gemmop.getA();
  auto B = gemmop.getB();
  auto C = gemmop.getC();

  int dByte = getDataBytes(A.getType());

  // Get shape info
  auto shapeA = trimTo2D(getShape(A.getType()));
  auto shapeB = trimTo2D(getShape(B.getType()));
  auto shapeC = trimTo2D(getShape(C.getType()));

  if(shapeC.size() == 1){
    llvm::SmallVector<int64_t, 2> newShapeC;
    newShapeC.push_back(shapeA[0]);
    newShapeC.push_back(shapeC[0]);
    shapeC = newShapeC;
  }


  // Extract M, N, K (assuming standard GEMM layout)
  int64_t M = shapeA[0];
  int64_t K = shapeA[1];
  int64_t N = shapeB[1];
  // sanity: C should be [M, N]
  assert(shapeC[0] == M && shapeC[1] == N && "C shape mismatch with A/B");

  // Prepare candidates
  struct Cand { 
    MatMulStrategy stationaryKind;
    int64_t T0, T1, row, col; 
    int64_t EC, TX, LAT; 
    // double util; 
    void dump() const{
      llvm::errs() << "[Cand] strategy=" 
        << getMethodStrRef(stationaryKind)
        << " T0=" << T0 << " T1=" << T1
        << " row=" << row << " col=" << col
        << " | EC=" << EC << " TX=" << TX
        << " LAT=" << LAT << "\n";
    }
  };
    
  std::optional<Cand> best;

  // Map rule for strategy (which dims correspond to row/col; which is innermost non-stationary)

  auto pairs = findFeasibleSpatialMap(num_pe, num_iob, /*limit=*/30000);

  std::vector<MatMulStrategy> strategies;
  if(_strategy == MatMulStrategy::Undefine){
    strategies.push_back(MatMulStrategy::InputStationary);
    strategies.push_back(MatMulStrategy::OutputStationary);
    strategies.push_back(MatMulStrategy::WeightStationary);
  }
  else{
    strategies.push_back(_strategy);
  }

  //// scan every(K, row, col)
  for(MatMulStrategy stationarykind : strategies){
    auto map = mappingFor(stationarykind, M, N, K);
    for (auto [rRaw,cRaw] : pairs) {
      // Basic feasibility: do not exceed mapped dims too much; allow <= mapped dim
      int64_t row = std::min<int64_t>(rRaw, map.Rmap);
      int64_t col = std::min<int64_t>(cRaw, map.Cmap);
      if (row <= 0 || col <= 0
        || !legalRowCol(stationarykind, std::make_pair(row, col), num_pe, num_iob)) 
        continue;

      /// scan T0 T1 under memory bound
      auto T1_cands = getAllDivisor(map.innermostNonStationary); 
      for (int T1: T1_cands) {
        for(int T0 = 1; T0 <= map.T0_dimension; T0++){
          if(T0 * T1 * dByte > bank_byte / 2)
            break;

          if(!CheckTileDivided(stationarykind, M, N, K, row, col, T0, T1)) 
            continue;

          // Compute EC (cycles)
          int64_t EC = execCycles(stationarykind, M, N, K, row, col, T0, T1, num_pe, num_iob);

          // Transfer volume (bytes) and time (cycles)
          int64_t TV = transferVolumn(stationarykind, M, N, K, row, col) * dByte;
          
          int64_t TX = ceilDiv(TV, std::max(1, BandByteWidth));  
          int64_t LAT = std::max<int64_t>(EC, TX);

          Cand cand{stationarykind, T0, T1, row, col, EC, TX, LAT};

          // Tie-break: smaller LAT; then smaller TV; then better alignment (higher util)
          auto better = [&](const Cand& a, const Cand& b){
            if ((double)abs(b.LAT - a.LAT)/(double)b.LAT <= (double)0.04) {
              return a.T1 >= b.T1;
            }
            if (a.LAT != b.LAT) return a.LAT < b.LAT;
            if (a.TX  != b.TX) return a.TX  < b.TX;
            return a.EC < b.EC;
          };
          cand.dump();
          if (!best || better(cand, *best)) best = cand;
        }
      }
    }
  }

  if (!best) {
    gemmop.emitError() << "[AutoSetDataflowStrategy] no feasible tiling found under "
                       << "PE/IOB/memory constraints.";
    return;
  }

  // ===== Attach the decision to the op =====
  const auto& ch = *best;
  ADORATensor::SystolicImplInterface Sinterface(gemmop);
  Sinterface.setStationaryKind(ch.stationaryKind);
  Sinterface.setTileSize(ArrayRef<int64_t>({ch.T0, ch.T1, ch.row, ch.col}));

  llvm::errs() << "Best strategy:";
  ch.dump();
}


}
}
} // namespace



void ADORAGemmOpStrategyDecisionPass::runOnOperation()
{
  mlir::ModuleOp m = getOperation();
  unsigned NumGPE = 0;
  unsigned NumIOB = 0; 
  unsigned bankBytes = 0;
  unsigned bandByteWidth = _bus_byte_band_width;

  if(_adg_fn != "-"){
    NumGPE = getInstanceNumFromADG(_adg_fn, "GPE");
    NumIOB = getInstanceNumFromADG(_adg_fn, "IOB");
    bankBytes = getIntegerAttrFromADG(_adg_fn, "iob_spad_bank_size");
  }
  else{
    NumGPE = _n_pes;
    NumIOB = _n_iobs;
    bankBytes = _sram_bank_kb_size * 1024;
  }

  MatMulStrategy stationary_kind = MatMulStrategy::Undefine;

  m.walk([&](mlir::ADORA::ADORATensor::GemmOp op) {
    ///// choose stationry kind
    if(_stationary_kind != "-"){
      if(_stationary_kind == "inputstationary"){
        stationary_kind = MatMulStrategy::InputStationary;
      }
      else if(_stationary_kind == "weightstationary"){
        stationary_kind = MatMulStrategy::WeightStationary;
      }
      else if(_stationary_kind == "outputstationary"){
        stationary_kind = MatMulStrategy::OutputStationary;
      }
      else{
        op.emitError() << "No this kind of stationary strategy."
                      <<"Please choose among \"inputstationary, weightstationary, outputstationary\".";
        return;
      }
    }


    /// Automatic set tile
    AutoSetDataflowStrategy(op, NumGPE, NumIOB, bankBytes, bandByteWidth, stationary_kind);

  });
}