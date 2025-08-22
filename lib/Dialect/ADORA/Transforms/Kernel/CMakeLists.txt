add_mlir_dialect_library(MLIRADORAKernelExtract
  AffineForToKernelPass.cpp
  KernelToFuncPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/IR
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/Transforms
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/Transforms/TaskGraph
  ${PROJ_INCLUDE_DIR}/lib/DFG/inc

  DEPENDS
  ADORAPassIncGen
  GeneralCDFGGen

  LINK_LIBS PUBLIC
  MLIRIR
  GeneralCDFGGen
  MLIRAffineDialect
  MLIRADORATransformsUtility
  # MLIRAffineToStandard
  # MLIRPass
  # MLIRTransforms
  # MLIRSCFDialect
  )