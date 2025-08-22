add_mlir_dialect_library(MLIRADORATransformsUtility
  Utility.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/IR
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/Transforms
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/Transforms/TaskGraph
  ${PROJ_INCLUDE_DIR}/lib/DFG/inc

  DEPENDS
  ADORAPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSCFDialect
  MLIRAffineDialect
  MLIRAffineToStandard
  )

set(LIBS
  MLIRADORATransformsUtility
)