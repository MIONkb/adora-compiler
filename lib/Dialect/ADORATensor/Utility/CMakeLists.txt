add_mlir_dialect_library(ADORATensorUtility
  Utility.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORATensor/IR

  DEPENDS
  MLIRFuncDialect
  MLIRMemRefDialect
  
  LINK_LIBS PUBLIC
  ADORATensorOps
  MLIRTensorDialect
  #MLIRIR
  #MLIRPass
  #MLIRTransforms
  #MLIRSCFDialect
  #MLIRAffineDialect
  #MLIRAffineToStandard
  )

set(LIBS
  ADORATensorUtility
)