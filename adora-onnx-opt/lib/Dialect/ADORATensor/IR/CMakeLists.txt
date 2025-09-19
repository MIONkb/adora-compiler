###### compile ADORATensor Dialect
add_mlir_dialect_library(ADORATensorOpsForONNX
  ${PROJECT_SOURCE_DIR}/lib/Dialect/ADORATensor/IR/Ops/OpHelper.cpp
  ${PROJECT_SOURCE_DIR}/lib/Dialect/ADORATensor/IR/Ops/MatMul.cpp
  ${PROJECT_SOURCE_DIR}/lib/Dialect/ADORATensor/IR/Ops/Gemm.cpp
  ${PROJECT_SOURCE_DIR}/lib/Dialect/ADORATensor/IR/ADORATensorDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORATensor/IR
  ${PROJECT_SOURCE_DIR}/include
  
  DEPENDS

  LINK_LIBS PUBLIC
  # ADORATensorInterface
  MLIRIR
  MLIRPass
  #MLIRTransforms
  #MLIRSCFDialect
  #MLIRAffineDialect
  #MLIRAffineToStandard
  )

set(LIBS
  ADORATensorOpsForONNX
)
