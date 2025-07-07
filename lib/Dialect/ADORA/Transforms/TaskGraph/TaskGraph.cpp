//===----------------------------------------------------------------------===//
// For Task Graph
//===----------------------------------------------------------------------===//
#include "RAAA/Dialect/ADORA/Transforms/TaskGraph/TaskGraph.h"

#include <iostream>
#include <fstream>
// #include <filesystem>
#include <string>

namespace mlir {
namespace ADORA {

/////////////////////////
/// TaskGraph class
/////////////////////////

/// @brief 
/// @param node 
/// @return 
static std::string getNodeTypeName(TaskNode* node){
  if(isa<KernelNode>(node))
    return "Kernel";
  else if(isa<BlockLoadNode>(node))
    return "BlockLoad";
  else if(isa<BlockStoreNode>(node))
    return "BlockStore";
  else 
    return "null";
}

/// @brief 
/// @param dep 
/// @return 
static std::string getDepTypeName(depType& dep){
  std::string deptypename;
  switch (dep)
  {
  case NoDep:
    deptypename = "NoDep";
    break;

  case Default:
    deptypename = "Default";
    break;

  case Depend:
    deptypename = "Depend";
    break;

  case Undefine:
  default:
    deptypename = "Undefine";
    break;
  }
  
  return deptypename;
}


/// @brief 
/// @param src 
/// @param dst 
/// @return 
static bool checkDefaultDependency(TaskNode* src, TaskNode* dst){
  if(isa<KernelNode>(src)){
    KernelNode* newsrc = dyn_cast<KernelNode>(src);
    if(isa<BlockStoreNode>(dst)){
      BlockStoreNode* newdst = dyn_cast<BlockStoreNode>(dst);
      if(newsrc->getKernelOp().hasKernelName()
          && newdst->getDataBlockStoreOp().hasKernelName()
          && newsrc->getKernelOp().getKernelName() == newdst->getDataBlockStoreOp().getKernelName()){
            return true;
      }
    }
  }
  else if(isa<BlockLoadNode>(src)){
    BlockLoadNode* newsrc = dyn_cast<BlockLoadNode>(src);
    if(isa<KernelNode>(dst)){
      KernelNode* newdst = dyn_cast<KernelNode>(dst);
      if(newsrc->getDataBlockLoadOp().hasKernelName()
          && newdst->getKernelOp().hasKernelName()
          && newsrc->getDataBlockLoadOp().getKernelName() == newdst->getKernelOp().getKernelName()){
            return true;
      }
    }
  }

  return false;
}


/// @brief add a node to graph, but do not analyze the dependency between this node and other nodes
/// @param node 
void TaskGraph::JustAddNode(TaskNode* node){
  int newIndex = getMaxNodeIdx() + 1;
  _nodes[node] = newIndex;
}

/// @brief add a node to graph and 
///        analyze the default dependency between this node and other nodes
///        Default dependendy is load-kernel-store dependency.  
/// @param node 
void TaskGraph::AddNodeAndAnalyzeDefaultDependency(TaskNode* newnode){
  JustAddNode(newnode);

  for (auto& pair : _nodes) {
    int index = pair.second;
    TaskNode* node = pair.first;

    if(checkDefaultDependency(/*src*/node, /*dst*/newnode)){
      addConnectionBetweenTwoNode(node, newnode, /*dep=*/depType::Default);
    }
    else if(checkDefaultDependency(/*src*/newnode, /*dst*/node)){
      addConnectionBetweenTwoNode(newnode, node, /*dep=*/depType::Default);
    }
  }
}

std::vector<TaskNode *> TaskGraph::getAllNodes(){
  std::vector<TaskNode *> result;
  for (const auto& pair : _nodes) {
    result.push_back(pair.first);
  }
  return result;
}

/// @brief print the graph to cout
void TaskGraph::dumpGraph() const{
  std::cout << "//==== Task Graph Dump ====//\n";
  
  for (const auto& pair : _nodes) {
    int index = pair.second;
    TaskNode* node = pair.first;
    std::cout << "//-------------------//\n";
    std::cout << "Node Index: " << index ;

    if(isa<KernelNode>(node))
      std::cout << ", KernelNode\n" ; 
    else if(isa<BlockLoadNode>(node))
      std::cout << ", BlockLoadNode\n" ; 
    else if(isa<BlockStoreNode>(node))
      std::cout << ", BlockStoreNode\n" ; 
    else
      std::cout << "\n" ;       

    std::cout << "  Operation: " ; 
    node->getOperation()->dump();

    std::cout << "   In Nodes: ";
    for (auto& inNode : node->getInNodes()) {
      std::cout << _nodes.at(inNode) << " "; // Assuming inNode is a pointer or you can add a method to print details
    }
    std::cout << "\n";
    
    std::cout << "   Out Nodes: ";
    for (auto& outNode : node->getOutNodes()) {
      std::cout << _nodes.at(outNode) << " "; // Assuming inNode is a pointer or you can add a method to print details
    }
    std::cout << "\n\n";
  }
}

/// @brief print the graph to filename as DOT style
/// @param filename the file address of the dot
void TaskGraph::dumpGraphAsDot(std::string& filename) const {
  std::ofstream ofs;
	ofs.open(filename.c_str());
  ofs << "Digraph G {\n";
  // std::string colors[4] = {"black", "purple", "blue", "yellow"};
  // nodes
	assert(_nodes.size() != 0);
  std::map<std::pair<TaskNode*, TaskNode*>, depType> edgestack; 
  // std::unordered_map<TaskNode*, std::string> nodeToName; 
  for(auto &elem : _nodes){
    int index = elem.second;
    TaskNode* node = elem.first;
    std::string NodeName = getNodeTypeName(node) + std::to_string(index);
    if(isa<KernelNode>(node)){
      KernelNode* kernelnode = dyn_cast<KernelNode>(node);
      ofs << NodeName << "[type = \"KernelNode\"";
      if(kernelnode->getKernelOp().hasKernelName()){
        ofs << ", KernelName = \"" << kernelnode->getKernelOp().getKernelName() <<"\"";
      }
      ofs << "];\n";
    }
    else if(isa<BlockLoadNode>(node)){
      BlockLoadNode* blockloadnode = dyn_cast<BlockLoadNode>(node);
      ofs << NodeName << "[type = \"BlockLoadNode\"";
      if(blockloadnode->getDataBlockLoadOp().hasKernelName()){
        ofs << ", KernelName = \"" << blockloadnode->getDataBlockLoadOp().getKernelName().str() <<"\"";
      }
      ofs << "];\n";
    }
    else if(isa<BlockStoreNode>(node)){
      BlockStoreNode* blockstorenode = dyn_cast<BlockStoreNode>(node);
      ofs << NodeName << "[type = \"BlockStoreNode\"";
      if(blockstorenode->getDataBlockStoreOp().hasKernelName()){
        ofs << ", KernelName = \"" << blockstorenode->getDataBlockStoreOp().getKernelName().str() <<"\"";
      }
      ofs << "];\n";
    }
    else{
      assert(false);
    }

    /// collect edges
    std::vector<TaskNode *> innodes = node->getInNodes();
    for(TaskNode * innode : innodes){
      edgestack[std::pair(innode, node)] = node->getInNodeDep(innode);
    }
  }

	// print edges
  // std::unordered_map<std::pair<TaskNode*, TaskNode*>, depType> edgestack; 
  for(auto &elem : edgestack){
    TaskNode* src = elem.first.first;
    TaskNode* dst = elem.first.second;
    depType dep = elem.second;
    std::string srcName = getNodeTypeName(src) + std::to_string(_nodes.at(src));
    std::string dstName = getNodeTypeName(dst) + std::to_string(_nodes.at(dst));
    
    ofs << srcName << " -> " << dstName 
        << "[color = black, style = bold, " 
        << "deptype = " << getDepTypeName(dep) << ", "
        << "label = \"deptype=" << getDepTypeName(dep) << "\"";

    ofs << "];\n";
  }
	ofs << "}\n";
	ofs.close();
}

int TaskGraph::getMaxNodeIdx(){
  int maxIndex = std::numeric_limits<int>::min();
  for (const auto& pair : _nodes) {
    if (pair.second > maxIndex) {
      maxIndex = pair.second;
    }
  }
  
  return (maxIndex == std::numeric_limits<int>::min()) ? -1 : maxIndex;
}

} // namespace ADORA
} // namespace mlir