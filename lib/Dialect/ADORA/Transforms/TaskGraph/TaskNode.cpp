//===----------------------------------------------------------------------===//
// For Task Graph
//===----------------------------------------------------------------------===//
#include "RAAA/Dialect/ADORA/Transforms/TaskGraph/TaskGraph.h"

namespace mlir {
namespace ADORA {



/////////////////////////////
// class TaskNode
/////////////////////////////
void TaskNode::addInNode(TaskNode* node){
  if(_innodes.count(node) == 0){
    /// is not an Input
    _innodes[node] = depType::Undefine;
  }
}

static depType judegeDep(depType dep1, depType dep2){
  switch (dep1)
  {
  case depType::Undefine :
    return dep2;
    break;

  case depType::Default :
    switch (dep2)
    {
      case depType::Undefine :
      return dep1;
      break;

      case depType::Default :
      return dep1;
      break;

      case depType::Depend :
      return depType::Depend;
      break;
    }
    break;

  case depType::Depend :
    return depType::Depend;
    break;

  default:
    return depType::Undefine;
  }
}

void TaskNode::addInNode(TaskNode* node, depType dep){
  if(_innodes.count(node) > 0){
    /// is not an Input
    _innodes[node] = judegeDep(dep, _innodes[node]);
  }
  else {
    _innodes[node] = dep;
  }
}

void TaskNode::delInNode(TaskNode* node){
  if(_innodes.count(node) > 0){
    _innodes.erase(node);
  }
}

depType TaskNode::getInNodeDep(TaskNode* node){
  if(_innodes.count(node) > 0)
    return _innodes[node];
  else 
    return depType::NoDep;
}

std::vector<TaskNode *> TaskNode::getInNodes(){
  std::vector<TaskNode *> keys;
  
  for (const auto& pair : _innodes) {
    keys.push_back(pair.first);
  }

  return keys;
}

void TaskNode::addOutNode(TaskNode* node){
  _outnodes.push_back(node);
}

void TaskNode::delOutNode(TaskNode* node){
  auto it = std::find(_outnodes.begin(), _outnodes.end(), node);
  if (it != _outnodes.end()) {
    _outnodes.erase(it); 
  }
}

std::vector<TaskNode *> TaskNode::getOutNodes(){
  return _outnodes;
}

//////////////////
/// KernelNode
//////////////////
bool KernelNode::classof(const TaskNode* node){
  if(isa<ADORA::KernelOp>(node->getOperation())){
    return true;
  }
  else{
    return false;
  }
}

//////////////////
/// BlockLoadNode
//////////////////
bool BlockLoadNode::classof(const TaskNode* node){
  if(isa<ADORA::DataBlockLoadOp>(node->getOperation())){
    return true;
  }
  else{
    return false;
  }
}

//////////////////
/// BlockLoadNode
//////////////////
bool BlockStoreNode::classof(const TaskNode* node){
  if(isa<ADORA::DataBlockStoreOp>(node->getOperation())){
    return true;
  }
  else{
    return false;
  }
}

} // namespace ADORA
} // namespace mlir