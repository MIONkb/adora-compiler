add_mlir_dialect_library(MLIRADORALoopTransforms
  AdjustMemoryFootprint.cpp
  AffineLoopReorder.cpp
  AffineLoopUnroll.cpp
  AffineLoopUnrollAndJam.cpp
  AutoDesignSpaceExplore.cpp
  AffineLoopSimplify.cpp
  AutoUnroll.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/IR
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/Transforms
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/Transforms/TaskGraph
  ${PROJ_INCLUDE_DIR}/lib/DFG/inc

  DEPENDS
  ADORAPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  GeneralCDFGGen
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRADORATransformsUtility
  # MLIRPass
  # MLIRTransforms
  # MLIRSCFDialect
  )

set(LIBS
  GeneralCDFGGen
)