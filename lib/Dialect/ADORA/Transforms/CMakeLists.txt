add_mlir_dialect_library(MLIRADORATransforms
  DFGgen.cpp
  AffineForToKernelPass.cpp
  AffineLoopSimplify.cpp
  KernelToFuncPass.cpp
  AutoDesignSpaceExplore.cpp
  AdjustMemoryFootprint.cpp
  SimplifyLoadStore.cpp
  AffineLoopUnroll.cpp
  AffineLoopUnrollAndJam.cpp
  AutoUnroll.cpp
  AffineLoopReorder.cpp
  DependencyAnalysis.cpp
  ScheduleAdoraTasks.cpp
  Utility.cpp
  TaskGraph/TaskNode.cpp
  TaskGraph/TaskGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/RAAA/Dialect/ADORA/IR
  ${PROJ_INCLUDE_DIR}/RAAA/Dialect/ADORA/Transforms
  ${PROJ_INCLUDE_DIR}/RAAA/Dialect/ADORA/Transforms/TaskGraph
  ${PROJ_INCLUDE_DIR}/lib/DFG/inc

  DEPENDS
  ADORAPassIncGen
  GeneralCDFGGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSCFDialect
  GeneralCDFGGen
  MLIRAffineDialect
  MLIRAffineToStandard
  )

set(LIBS
  GeneralCDFGGen
)