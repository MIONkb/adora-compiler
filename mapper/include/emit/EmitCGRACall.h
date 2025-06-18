#ifndef CGRV_EMITCGRACALL_H
#define CGRV_EMITCGRACALL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/raw_ostream.h"

#include "RAAA/Dialect/ADORA/IR/ADORA.h"

#include "mapper/mapper.h"
#include "mapper/mapper_sa.h"
#include "mapper/io_scheduler.h"


using namespace mlir;
using namespace mlir::ADORA;

namespace mlir {
namespace ADORA {
void SimplifyBlockAccessOp(mlir::ModuleOp m);
} /// ADORA
} /// mlir


class Op_Name_C{
public:
  int id;
  std::string type;
  std::string name(){return type + "_" + std::to_string(id);}
  Op_Name_C(std::string type, int id): type(type) , id(id) {}
  Op_Name_C(int id, std::string type): type(type) , id(id) {}
  Op_Name_C(){}
};

// class CGRVOpEmitter;

class CGRACallEmitter {
public:
  void emitFunctionHead(func::FuncOp &funcop, llvm::raw_ostream &os);
  void emitBlock(mlir::Block &block, llvm::raw_ostream &os);
  void DataBlockOperationsToSPADInfo(mlir::ADORA::KernelOp& kernel, MapperSA* mapper);
  std::string GenerateCGRAConfig(ADORA::KernelOp& kernel, MapperSA* mapper);
  std::string GenerateCGRAConfig(ADORA::KernelOp& kernel, Configuration configuration, ADG* adg);
  void GenerateCGRACFGAndEXE(ADORA::KernelOp& kernel, MapperSA* mapper);
  void GenerateCGRACFGAndEXE(ADORA::KernelOp& kernel, Configuration configuration, ADG* adg);
  bool emitCGRACallFunction(llvm::raw_ostream &os);

  llvm::SmallDenseMap<mlir::Value, Op_Name_C> getValueNameList(){return _value_name_list;}
  void appendValueNameList(mlir::Value v, Op_Name_C info){_value_name_list[v] = info;}
  std::string lookupName(mlir::Value v){ 
    if(_value_name_list.count(v))
      return _value_name_list[v].name();
    else
      return "";
  }

  std::string lookupVarConfigName(const std::string);

  CGRACallEmitter(mlir::ModuleOp m): _moduleop(m){}
  unsigned getIndent(){return _currentIndent;}

  friend class CGRVOpEmitter;
  // CGRVOpEmitter opEmitter;
  /// Kernel and its configuration
  llvm::SmallDenseMap<ADORA::KernelOp, std::pair<std::string, int>> KnToCfgArrayInfo;   // map : kernelop -> (cfg array's name, cfgnum)
  llvm::SmallDenseMap<ADORA::KernelOp, std::string> KnToCfgData; // map : kernelop -> total cfg array declaration
  llvm::SmallDenseMap<ADORA::KernelOp, std::string> KnToCfgExe;

  llvm::SmallVector<dfgIoInfo> getDfgIoInfosFromBlockLoad(ADORA::DataBlockLoadOp op) {return _LoadToDfgIoInfos[op];}
  dfgIoInfo getDfgIoInfosFromBlockStore(ADORA::DataBlockStoreOp op) {return _StoreToDfgIoInfo[op];}

  llvm::SmallVector<std::pair<int, dfgIoInfo>> getSPMInfosFromBlockLoad(ADORA::DataBlockLoadOp op) {return _LoadToSPMInfos[op];}
  std::pair<int, dfgIoInfo> getSPMInfosFromLocalAlloc(ADORA::LocalMemAllocOp op) {return _LocalAllocToSPMInfo[op];}
  

  // @brief Establishes placement constraints for local memory allocations within a given kernel operation.
  /// @param kernel A reference to an ADORA::KernelOp object representing the kernel operation.
  /// @param mapper A pointer to a MapperSA object used for accessing the data flow graph (DFG) and architecture description graph (ADG).
  void preestablishPlacementConstraints(ADORA::KernelOp& kernel, MapperSA* mapper);

  llvm::SmallDenseMap<ADORA::KernelOp, Configuration> KnToConfiguration;
  void setMapResult(ADORA::KernelOp k, MapperSA* mapper);

  void setADG(ADG* _) {_adg = _;}
  ADG* getADG(){return _adg;}

private:
  mlir::ModuleOp _moduleop;
  std::stringstream _CFGandEXE;
  std::map<KernelOp, std::map<int, dfgIoInfo>> _kernel_to_dfg_io_infos;
  // std::map<int, int> _dfgIoSpadAddrs;
  ADG* _adg;

  std::map<KernelOp, uint64_t> _kernel_to_iob_ens;
  // uint64_t _iob_ens = 0;
  llvm::SmallDenseMap<ADORA::DataBlockLoadOp, llvm::SmallVector<dfgIoInfo>> _LoadToDfgIoInfos;
  llvm::SmallDenseMap<ADORA::DataBlockStoreOp, dfgIoInfo> _StoreToDfgIoInfo;

  llvm::SmallDenseMap<ADORA::DataBlockLoadOp, llvm::SmallVector< std::pair<int, dfgIoInfo> > > _LoadToSPMInfos; // int : spm bank idx, dfgioinfo
  llvm::SmallDenseMap<ADORA::LocalMemAllocOp, std::pair<int, dfgIoInfo>> _LocalAllocToSPMInfo;

  llvm::SmallDenseMap<mlir::Value, Op_Name_C> _value_name_list;

  /// Variables for emitting
  unsigned _currentIndent = 0;
  void addIndent(){_currentIndent += 2;}
  void reduceIndent(){_currentIndent = _currentIndent >= 2 ? _currentIndent - 2 : 0;}
};

namespace mlir {
namespace ADORA {




} // namespace ADORA
} // namespace mlir

#endif // CGRV_EMITCGRACALL_H
