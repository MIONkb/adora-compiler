add_mlir_dialect_library(MLIRADORAOps
  ADORADialect.cpp
  ADORAOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/RAAA/Dialect/ADORA

  DEPENDS
  MLIRADORAOpInterfacesIncGen
  MLIRADORAOpsEnumsGen
  MLIRADORAOpsAttributesIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRTensorDialect
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRFuncDialect
  MLIRViewLikeInterface
  )
