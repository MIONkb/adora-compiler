#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Support/FileUtilities.h"

#include "RAAA/Dialect/ADORA/IR/ADORA.h"
#include "RAAA/Dialect/ADORA/Transforms/Passes.h"
#include "./PassDetail.h"

using namespace mlir;
using namespace mlir::ADORA;
using namespace mlir::affine;

namespace {

// class AffineForSimplifyConverter : public OpRewritePattern<affine::AffineForOp> {
// public:
//   using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;
    
//   LogicalResult matchAndRewrite(affine::AffineForOp forop,
//                                   PatternRewriter &rewriter) const final {
//     mlir::Block* parentBlock = forop->getBlock();
//     rewriter.updateRootInPlace(forop, [&]() {
//       Location loc = forop.getLoc();
//       int tc = getConstantTripCount(forop).value_or(0);
//       if(tc == 1 && forop.getNumResults() == 0){
//         AffineBound lb = forop.getLowerBound();
//         AffineMap lbMap = lb.getMap();
//         if(lb.getNumOperands() == 0 
//             && lbMap.getResults().size() == 1
//             && lbMap.getResult(0).getKind() == AffineExprKind::Constant){
//           /// replace the loop iter var with a constant
//           int lb_value = lbMap.getResult(0).dyn_cast<AffineConstantExpr>().getValue();
//           mlir::Value constlb = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(lb_value));
//           forop.getInductionVar().replaceAllUsesWith(constlb);
          
//           Block* loopBlock = forop.getBody();
//           // assert(isa<AffineYieldOp>(loopBlock->getTerminator()));
//           // loopBlock->getTerminator()->erase();
    
//           /// move the body of the loop to the parent block
//           // mlir::Region* parentRegion = forop->getBlock()->getParent();
    
//           /// Move the loop body to the parent block, placing it before the forOp.
//           mlir::SmallVector<mlir::Operation*> ops_tomove;
//           for(mlir::Operation& operation : *loopBlock){
//             if(&operation == forop.getOperation())
//               break;
//             if(&operation == loopBlock->getTerminator())
//               continue;
//             ops_tomove.push_back(&operation);
//           }
    
//           for(auto operation : ops_tomove){
//             operation->moveBefore(forop);
//           }
//           // forop->getBlock()->dump();
//           /// Erase the forOp itself.
//           forop->erase();
    
//           /// simplify affine maps
//           // simplifyLoadAndStoreOpsInRegion(*parentRegion);
  
//           // parentRegion->dump();

//         }
//       }
//     });
//     parentBlock->dump();
//     return success();
//   }
// };
    
/// A pass to lower math operations 
struct SimplifyAffineLoopLevels
    : public SimplifyAffineLoopLevelsBase<SimplifyAffineLoopLevels> {
  SimplifyAffineLoopLevels() = default;

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    // RewritePatternSet patterns(&getContext());
    // ConversionTarget target(getContext());
    // GreedyRewriteConfig config;
    // config.maxIterations = 1;

    // patterns.add< 
    //   AffineForSimplifyConverter
    // >(patterns.getContext());

    // m.dump();
    // if (failed(applyPartialConversion(getOperation(), target,
    //                                   std::move(patterns))))
    //   signalPassFailure();
    // // if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns), config)))
    // //   signalPassFailure();
    
    // m.dump();
    mlir::SmallVector<AffineForOp> to_erase;
    m.walk([&](AffineForOp forop) {
      mlir::OpBuilder b(forop.getOperation());
      Location loc = forop.getLoc();
      int tc = getConstantTripCount(forop).value_or(0);
      if(tc == 1 && forop.getNumResults() == 0){
        AffineBound lb = forop.getLowerBound();
        AffineMap lbMap = lb.getMap();
        if(lb.getNumOperands() == 0 
            && lbMap.getResults().size() == 1
            && lbMap.getResult(0).getKind() == AffineExprKind::Constant){
          /// replace the loop iter var with a constant
          int lb_value = lbMap.getResult(0).dyn_cast<AffineConstantExpr>().getValue();
          mlir::Value constlb = b.create<arith::ConstantOp>(loc, b.getIndexAttr(lb_value));
          forop.getInductionVar().replaceAllUsesWith(constlb);
          
          Block* loopBlock = forop.getBody();
          // assert(isa<AffineYieldOp>(loopBlock->getTerminator()));
          // loopBlock->getTerminator()->erase();
    
          /// move the body of the loop to the parent block
          // mlir::Region* parentRegion = forop->getBlock()->getParent();
    
          /// Move the loop body to the parent block, placing it before the forOp.
          mlir::SmallVector<mlir::Operation*> ops_tomove;
          for(mlir::Operation& operation : *loopBlock){
            if(&operation == forop.getOperation())
              break;
            if(&operation == loopBlock->getTerminator())
              continue;
            ops_tomove.push_back(&operation);
          }
    
          for(auto operation : ops_tomove){
            operation->moveBefore(forop);
          }
          // forop->getBlock()->dump();
          /// Erase the forOp itself.
          to_erase.push_back(forop);
    
          /// simplify affine maps
          // simplifyLoadAndStoreOpsInRegion(*parentRegion);
  
          // parentRegion->dump();

        }
      }
    });

    for(AffineForOp& op : to_erase)
      op.erase();
  }
};


} // end anonymous namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> 
  mlir::ADORA::createSimplifyAffineLoopLevelsPass() {
  return std::make_unique<SimplifyAffineLoopLevels>();
}