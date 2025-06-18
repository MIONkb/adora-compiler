add_library(GeneralCDFGGen
  mlir_cdfg.cpp
  mlir_cdfg_node.cpp
  llvm_cdfg.cpp
  llvm_cdfg_node.cpp
  llvm_cdfg_edge.cpp
  )


# set(LLVM_INCLUDE_DIRS "/home/tianyi/MLIRCGRA/llvm-project/build/include")
# set(LLVM_LIBRARY_DIRS "/home/tianyi/MLIRCGRA/llvm-project/build/lib")

target_include_directories(GeneralCDFGGen
  PRIVATE ${PROJ_INCLUDE_DIR}/lib/DFG/inc
  PRIVATE ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(GeneralCDFGGen
  PUBLIC MLIRIR
  PUBLIC MLIRPass
  PUBLIC MLIRTransforms
  PUBLIC MLIRSCFDialect
  PRIVATE ${LLVM_LIBRARY_DIRS}
)

add_dependencies(GeneralCDFGGen ADORAPassIncGen)