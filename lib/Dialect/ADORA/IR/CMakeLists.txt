add_subdirectory(KernelOp)
add_mlir_dialect_library(MLIRADORAOps
  ADORADialect.cpp
  ADORAOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA

  DEPENDS
  MLIRADORAOpsEnumsGen
  MLIRADORAOpsAttributesIncGen
  MLIRADORAOpInterfacesIncGen
  MLIRTensorDialect
  MLIRFuncDialect


  LINK_COMPONENTS
  Core
  
  LINK_LIBS PUBLIC
  MLIRADORAKernelOp
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRDialectUtils
  MLIRIR
  MLIRViewLikeInterface
  MLIRSideEffectInterfaces
  )

