# To solve error: exception handling disabled, use '-fexceptions' to enable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPDLOG_NO_EXCEPTIONS ")
set(SPDLOG_NO_EXCEPTIONS ON)
add_mlir_dialect_library(ADORATensorTransforms
  LinalgToSystolicGEMM.cpp
  TensorCDFGGenPass.cpp
  GemmOpStrategyDecision.cpp
  GetKernelsInTensorFunctions.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORATensor/IR

  DEPENDS
  ADORATensorPassIncGen
  
  LINK_LIBS PUBLIC
  GeneralCDFGGen
  MLIRADORALoopTransforms
  ADORATensorOps
  ADORATensorLower
  ADORATensorUtility
  MLIRTensorDialect
  #MLIRIR
  #MLIRPass
  #MLIRTransforms
  #MLIRSCFDialect
  #MLIRAffineDialect
  #MLIRAffineToStandard
  #ADORAMapTensorOp
  )

set(LIBS
  ADORATensorTransforms
)

