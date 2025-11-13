add_mlir_dialect_library(MLIRADORAKernelOp
  KernelOp.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA

  # DEPENDS
  # MLIRADORAOpsEnumsGen
  # MLIRADORAOpsAttributesIncGen
  # MLIRADORAOpInterfacesIncGen
  # MLIRTensorDialect
  # MLIRFuncDialect


  # LINK_COMPONENTS
  # Core

  LINK_LIBS PUBLIC
  # MLIRSupport
  MLIRInferTypeOpInterface
  # MLIRDialectUtils
  # MLIRIR
  # MLIRViewLikeInterface
  # MLIRSideEffectInterfaces
  )
