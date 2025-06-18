#ifndef __DFG_EDGE_H__
#define __DFG_EDGE_H__

#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <assert.h>
#include "graph/graph_edge.h"
#include "spdlog/spdlog.h"
#include "../../lib/DFG/inc/llvm_cdfg_edge.h"
// #include "mlir_cdfg.h"
// enum EdgeType{ 
//     EDGE_TYPE_DATA, // data dependence
//     EDGE_TYPE_CTRL, // control dependence
//     EDGE_TYPE_MEM,  // loop-carried memory dependence 
// };

class DFGEdge : public GraphEdge
{
private:
    EdgeType _type = EDGE_TYPE_DATA;
    bool _backedge = false;
    //int _logicLat = 0; // due to multport add a logic lat //mulrport     
    int _iterDist = 0; // iteration distance for loop-carried dependence    
public:
    using GraphEdge::GraphEdge; // C++11, inherit parent constructors
    EdgeType type(){ return _type; }
    void setType(EdgeType type){ _type = type; }
    bool isMemEdge(){ return _type == EDGE_TYPE_MEM; }
    bool isMemBackEdge(){ return _type == EDGE_TYPE_MEM && _backedge;}
    bool isBackEdge(){ return _backedge; }
    void setBackEdge(bool back){ _backedge = back; }
    // void setlogicLat(int logicLat){ _logicLat = logicLat; }
    // int logicLat(){ return _logicLat; }
    void setIterDist(int dist){ _iterDist = dist; }
    int iterDist(){ return _iterDist; }
};





#endif