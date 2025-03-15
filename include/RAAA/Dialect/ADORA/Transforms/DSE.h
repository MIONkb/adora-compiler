//===- DSE.h --------------------- --------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
#ifndef ADORA_CGRA_OPT_DSE_H
#define ADORA_CGRA_OPT_DSE_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"


#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"

#include "RAAA/Dialect/ADORA/IR/ADORA.h"
#include "RAAA/Dialect/ADORA/Transforms/Passes.h"
// #include "./PassDetail.h"

using namespace std;
using namespace mlir;
using namespace mlir::affine;
using namespace llvm;

namespace mlir {
namespace ADORA {
using DesignPoint = SmallVector<unsigned,12>;
enum UnrollStrategy {Undecided, Unroll, Unroll_and_Jam, CannotUnroll};

class ForNode
{
private:
  /* data */
  mlir::affine::AffineForOp ForOp;
  ForNode* ParentNode = nullptr;
  llvm::SmallVector<ForNode*> ChildrenNodes;
  int Level;
  UnrollStrategy _UnrollStrategy = UnrollStrategy::Undecided;
  bool _IsOutOfKernel = false;

public:
  /*** Type define ***/
  using FactorList = SmallVector<unsigned>;

  /* tiling factor */
  FactorList TilingFactors;
  /* unroll factor */
  FactorList UnrollFactors;

  /// builder function for class ADORA::ForNode
  ForNode(AffineForOp For): ForOp(For){};
  ForNode(AffineForOp For, unsigned level): ForOp(For), Level(level){};
  // ~ForNode();

  mlir::affine::AffineForOp& getForOp() {return ForOp;}

  void setChildren(llvm::SmallVector<ForNode*>& Children){ ChildrenNodes = Children;}
  llvm::SmallVector<ForNode*> getChildren() const {return ChildrenNodes;}

  void setParent(ForNode* Parent){ ParentNode = Parent;}
  ForNode* getParent() const {return ParentNode;}

  void setLevel(const int l){ Level = l;}
  int getLevel() const {return Level;}

  void setUnrollStrategy(const UnrollStrategy _){ _UnrollStrategy = _;}
  UnrollStrategy getUnrollStrategy() const {return _UnrollStrategy;}

  bool IsInnermost();
  bool HasParentFor(){ return ParentNode != nullptr;}
  bool IsThisLevelPerfect();

  bool IsOutOfKernel(){return _IsOutOfKernel;}
  void setOutOfKernel(bool _ = true){_IsOutOfKernel = _;}
  //////////////
  /// dump functions 
  //////////////
  void dumpNode(){
    if(IsOutOfKernel()){
      llvm::errs() << Level << ":"<< " Out of kernel.";     
    }
    else {
      for(int cnt = 0; cnt < Level; cnt++)
        llvm::errs() << " ";     
      llvm::errs() << Level << ":";
      if(!HasParentFor())
        llvm::errs() << " outermost";
      else{
        if(IsThisLevelPerfect())
          llvm::errs() << " perfect";
        else 
          llvm::errs() << " imperfect";
      }

      if(IsInnermost())
        llvm::errs() <<", innermost";

      if(UnrollFactors.size()!=0){
        llvm::errs() <<", unroll factor: ";
        for(auto ft : UnrollFactors)
          llvm::errs() << ft << " ";
      }    
    }

    llvm::errs() << "\n";
  }
  void dumpForOp(){
    this->getForOp().dump();
  }
  void dumpTree(){
    // dumpForOp();
    for(ForNode* Child : ChildrenNodes){
      Child->dumpTree();
    }

    dumpNode();
  }
  ForNode* getOutermostFor() {
    if(HasParentFor()){
      return getParent()->getOutermostFor();
    }
    else
      return this;
  }

  unsigned getMaxUnrollFactor(){
    unsigned UF_max = 0;
    for(auto UF: UnrollFactors)
      UF_max = UF_max > UF ? UF_max : UF;
    return UF_max;
  }

  bool operator==(const ForNode &RHS) const {
    return this->ForOp == RHS.ForOp && 
      this->ParentNode == RHS.ParentNode && 
      this->ChildrenNodes == RHS.ChildrenNodes && 
      this->Level == RHS.Level;
  }

  bool operator!=(const ForNode &RHS) const {
    return !(*this == RHS);
  }

};
using UnrollDesignPoint = SmallVector<unsigned>;
SmallVector<DesignPoint> ExpandTilingAndUnrollingFactors(ForNode, SmallVector<DesignPoint>);
SmallVector<unsigned> FindUnrollingFactors(ADORA::ForNode& Node);
SmallVector<ADORA::ForNode> createAffineForTree(func::FuncOp topfunc);
SmallVector<ADORA::ForNode> createAffineForTreeInsideKernel(ADORA::KernelOp kernel);
SmallVector<ADORA::ForNode> createAffineForTreeAroundKernel(ADORA::KernelOp kernel); /// Contains inside loops and one level of parent op
ADORA::ForNode* findTargetLoopNode(SmallVector<ADORA::ForNode>& NodeVec, mlir::affine::AffineForOp forop);
void NestedGenTree(ADORA::ForNode* rootNode, SmallVector<ADORA::ForNode>& NodeVec);
SmallVector<SmallVector<unsigned>> ConstructUnrollSpace(SmallVector<ADORA::ForNode> ForNodes);

//// Self-defined unroll 
LogicalResult loopUnrollByFactor_opt(AffineForOp forOp, uint64_t unrollFactor, bool cleanUpUnroll);
LogicalResult loopUnrollAndJamByFactor(affine::AffineForOp& forop, unsigned ur_factor);
// class ADG : public Graph
// {
// private:
//     int _numGpeNodes;
//     int _numIobNodes;
//     // int _cfgDataWidth;
//     // int _cfgAddrWidth;
//     // int _cfgBlkOffset;
//     // int _cfgSpadDataWidth; // data width of the config spad
//     // int _loadLatency;
//     // int _storeLatency;
//     // int _cfgSpadSize; // size of scratchpad for config
//     // int _iobAgNestLevels; // AG nested levels in IOB
//     int _iobSpadBankSize; // size of each scratchpad bank for IOB
//     // std::map<int, std::vector<int>> _iobToSpadBanks; // the scratchpad banks connected to each IOB, <iob-index, <banks>>
//     // std::vector<uint64_t> _cfgBits;

//     // std::map<int, ADGNode*> _nodes;   // <node-id, node>
//     // std::map<int, ADGEdge*> _edges;   // <edge-id, edge>

//     ADG(const ADG&) = delete; // disable the default copy construct function

// public:
//     ADG();
//     ~ADG();
//     int numGpeNodes(){ return _numGpeNodes; }
//     void setNumGpeNodes(int numGpeNodes){ _numGpeNodes = numGpeNodes; }
//     int numIobNodes(){ return _numIobNodes; }
//     void setNumIobNodes(int numIobNodes){ _numIobNodes = numIobNodes; }
//     // int cfgDataWidth(){ return _cfgDataWidth; }
//     // void setCfgDataWidth(int cfgDataWidth){ _cfgDataWidth = cfgDataWidth; }
//     // int cfgAddrWidth(){ return _cfgAddrWidth; }
//     // void setCfgAddrWidth(int cfgAddrWidth){ _cfgAddrWidth = cfgAddrWidth; }
//     // int cfgBlkOffset(){ return _cfgBlkOffset; }
//     // void setCfgBlkOffset(int cfgBlkOffset){ _cfgBlkOffset = cfgBlkOffset; }
//     // int cfgSpadDataWidth(){ return _cfgSpadDataWidth; }
//     // void setCfgSpadDataWidth(int cfgSpadDataWidth){ _cfgSpadDataWidth = cfgSpadDataWidth; }
//     // int loadLatency(){ return _loadLatency; }
//     // void setLoadLatency(int lat){ _loadLatency = lat; }
//     // int storeLatency(){ return _storeLatency; }
//     // void setStoreLatency(int lat){ _storeLatency = lat; }
//     // int iobAgNestLevels(){ return _iobAgNestLevels; }
//     // void setIobAgNestLevels(int levels){ _iobAgNestLevels = levels; }
//     // int cfgSpadSize(){ return _cfgSpadSize; }
//     // void setCfgSpadSize(int size){ _cfgSpadSize = size; }
//     int iobSpadBankSize(){ return _iobSpadBankSize; }
//     void setIobSpadBankSize(int size){ _iobSpadBankSize = size; }
//     // const std::map<int, std::vector<int>>& iobToSpadBanks(){ return _iobToSpadBanks; }
//     // const std::vector<int>& iobToSpadBanks(int iobId){ return _iobToSpadBanks[iobId]; }
//     // void setIobToSpadBanks(int iobId, std::vector<int> banks){ _iobToSpadBanks[iobId] = banks; }

//     // const std::map<int, ADGNode*>& nodes(){ return _nodes; }
//     // const std::map<int, ADGEdge*>& edges(){ return _edges; }
//     // ADGNode* node(int id);
//     // ADGEdge* edge(int id);
//     // void addNode(int id, ADGNode* node);
//     // void addEdge(int id, ADGEdge* edge);
//     // void delNode(int id);
//     // void delEdge(int id);

//     // ADG& operator=(const ADG& that);

//     // void print();
    
// };


} // namespace ADORA
} // namespace mlir



#endif //ADORA_CGRA_OPT_DSE_H