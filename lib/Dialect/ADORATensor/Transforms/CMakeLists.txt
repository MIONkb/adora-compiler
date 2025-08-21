add_mlir_dialect_library(ADORATensorTransforms
  LinalgToSystolicGEMM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/RAAA/Dialect/ADORATensor/IR

  DEPENDS
  ADORATensorPassIncGen
  
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
  ADORATensorTransforms
)