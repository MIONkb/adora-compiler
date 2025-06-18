//===----------------------------------------------------------------------===//
// For Task Graph
//===----------------------------------------------------------------------===//
#include "RAAA/Dialect/ADORA/Transforms/TaskGraph/TaskGraph.h"
#include "RAAA/Dialect/ADORA/Transforms/DependencyAnalysis.h"

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
  else if(isa<LocalAllocNode>(node))
    return "LocalAlloc";
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

  case SourceToStore:
    deptypename = "SourceToStore";
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
static depType checkDefaultDependency(TaskNode* src, TaskNode* dst){
  if(isa<KernelNode>(src)){
    KernelNode* newsrc = dyn_cast<KernelNode>(src);
    if(isa<BlockStoreNode>(dst)){
      BlockStoreNode* newdst = dyn_cast<BlockStoreNode>(dst);
      if(newsrc->getKernelOp().hasKernelName()
          && newdst->getDataBlockStoreOp().hasKernelName()
          && newsrc->getKernelOp().getKernelName() == newdst->getDataBlockStoreOp().getKernelName()){
            return depType::Default;
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
            return depType::Default;
      }
    }
    else if(isa<BlockStoreNode>(dst)){
      BlockStoreNode* newdst = dyn_cast<BlockStoreNode>(dst);
      if(newsrc->getDataBlockLoadOp().hasKernelName()
          && newdst->getDataBlockStoreOp().hasKernelName()
          && newsrc->getDataBlockLoadOp().getKernelName() == newdst->getDataBlockStoreOp().getKernelName()
          && newsrc->getDataBlockLoadOp().getId() == newdst->getDataBlockStoreOp().getId()){
            return depType::SourceToStore;
      }
    }
  }
  
  else if(isa<LocalAllocNode>(src)){
    LocalAllocNode* newsrc = dyn_cast<LocalAllocNode>(src);
    if(isa<BlockStoreNode>(dst)){
      BlockStoreNode* newdst = dyn_cast<BlockStoreNode>(dst);
      if(newsrc->getLocalMemAllocOp().hasKernelName()
          && newdst->getDataBlockStoreOp().hasKernelName()
          && newsrc->getLocalMemAllocOp().getKernelName() == newdst->getDataBlockStoreOp().getKernelName()
          && newsrc->getLocalMemAllocOp().getId() == newdst->getDataBlockStoreOp().getId()){
            return depType::SourceToStore;
      }
    }
    if(isa<KernelNode>(dst)){
      KernelNode* newdst = dyn_cast<KernelNode>(dst);
      if(newsrc->getLocalMemAllocOp().hasKernelName()
          && newdst->getKernelOp().hasKernelName()
          && newsrc->getLocalMemAllocOp().getKernelName() == newdst->getKernelOp().getKernelName()){
            return depType::Default;
      }
    }
  }

  return depType::Undefine;
}








////////////////////////////////////////////
/// TaskGraph class
////////////////////////////////////////////
/// @brief add a node to graph, but do not analyze the dependency between this node and other nodes
/// @param node 
void TaskGraph::JustAddNode(TaskNode* node){
  int newIndex = getMaxNodeIdx() + 1;
  _nodes[node] = newIndex;
}

/// @brief delete a node from graph, but do not delete the operation of the node
/// @param node 
void TaskGraph::JustDeleteNode(TaskNode* node){
  auto it = _nodes.find(node);
  if (it != _nodes.end()) {
    int index = it->second;
    _nodes.erase(it);
  }
}

/// @brief delete a node from graph and delete the operation of the node in mlir module
/// @param node 
void TaskGraph::DeleteNodeOperation(TaskNode* node){
  JustDeleteNode(node);
  node->delNodeOperation();
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

    if(checkDefaultDependency(/*src*/node, /*dst*/newnode) == depType::Default){
      addConnectionBetweenTwoNode(node, newnode, /*dep=*/depType::Default);
    }
    else if(checkDefaultDependency(/*src*/newnode, /*dst*/node) == depType::Default){
      addConnectionBetweenTwoNode(newnode, node, /*dep=*/depType::Default);
    }
    else if(checkDefaultDependency(/*src*/node, /*dst*/newnode) == depType::SourceToStore){
      addConnectionBetweenTwoNode(node, newnode, /*dep=*/depType::SourceToStore);
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

/// @brief Retrieves the TaskNode associated with the specified Operation.
/// @param op A pointer to the mlir::Operation to find the associated TaskNode for.
/// @return A pointer to the matching TaskNode, or nullptr if no match is found.
TaskNode* TaskGraph::getNode(mlir::Operation* op) {
    // Iterate through each TaskNode stored in _nodes
    // op->dump();
    // std::cout << op << "\n";
    for (const auto& pair : _nodes) {
        TaskNode* node = pair.first; // Get the TaskNode
        mlir::Operation* nodeOp = node->getOperation(); // Get the corresponding operation

        // nodeOp->dump();
        // std::cout << nodeOp << "\n";

        // Check if the current node's operation matches the given operation
        if (nodeOp == op) {
          return node; // Found a matching node, return it
        }
    }
    
    // Return nullptr if no matching node is found
    return nullptr;
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
    else if(isa<LocalAllocNode>(node))
      std::cout << ", LocalAllocNode\n" ; 
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
    else if(isa<LocalAllocNode>(node)){
      LocalAllocNode* localallocnode = dyn_cast<LocalAllocNode>(node);
      ofs << NodeName << "[type = \"LocalAllocNode\"";
      if(localallocnode->getLocalMemAllocOp().hasKernelName()){
        ofs << ", KernelName = \"" << localallocnode->getLocalMemAllocOp().getKernelName().str() <<"\"";
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
    
    if(dep == depType::SourceToStore){
      ofs << srcName << " -> " << dstName 
          << "[color = blue, style = bold, " 
          << "deptype = " << getDepTypeName(dep) << ", "
          << "label = \"deptype=" << getDepTypeName(dep) << "\"";
    }
    else{
      ofs << srcName << " -> " << dstName 
          << "[color = black, style = bold, " 
          << "deptype = " << getDepTypeName(dep) << ", "
          << "label = \"deptype=" << getDepTypeName(dep) << "\"";
    }


    ofs << "];\n";
  }
	ofs << "}\n";
	ofs.close();
}

/// @brief print the node to cout
void TaskGraph::dumpNode(TaskNode* node) const{
    if(_nodes.count(node) == 0){
      std::cout << "This node is not in graph.";
      return;      
    }

    std::cout << "Node Index: " << _nodes.at(node);

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
    std::cout << "\n";
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