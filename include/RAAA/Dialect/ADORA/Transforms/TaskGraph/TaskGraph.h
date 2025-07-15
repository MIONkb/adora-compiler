//===----------------------------------------------------------------------===//
// For Task Graph
//===----------------------------------------------------------------------===//
#ifndef ADORA_TASK_GRAPH_H
#define ADORA_TASK_GRAPH_H
#include "RAAA/Dialect/ADORA/IR/ADORA.h"
#include "TaskNode.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include <unordered_map>
#include <vector>

namespace mlir {
namespace ADORA {
/////////////////////////
/// base graph class
/////////////////////////
class TaskGraph{
private:
  std::unordered_map<TaskNode *, int> _nodes;  
  mlir::Operation* _parentOp;  
  
public:
  void JustAddNode(TaskNode* node);
  void JustDeleteNode(TaskNode* node);
  void DeleteNodeOperation(TaskNode* node);

  void AddNodeAndAnalyzeDefaultDependency(TaskNode* node);
  template <typename T> void AddNodeAndAnalyzeDefaultDependency(T* node){
    AddNodeAndAnalyzeDefaultDependency(dyn_cast<TaskNode>(node));
  };

  void setParentOp(mlir::Operation* op){_parentOp = op;}
  mlir::Operation* getParentOp(){return _parentOp;}

  std::vector<TaskNode *> getAllNodes();
  TaskNode* getNode(mlir::Operation* op);

  void dumpGraph() const;
  void dumpGraphAsDot(std::string& filename) const;
  void dumpNode(TaskNode* node) const;
private:
  int getMaxNodeIdx();

public:
  TaskGraph(){}
  ~TaskGraph(){}
};

//////////////////
/// Some tool functions
//////////////////


} // namespace ADORA
} // namespace mlir
#endif