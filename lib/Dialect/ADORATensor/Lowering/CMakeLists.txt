# To solve error: exception handling disabled, use '-fexceptions' to enable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPDLOG_NO_EXCEPTIONS ")
set(SPDLOG_NO_EXCEPTIONS ON)

add_mlir_dialect_library(ADORATensorLower
  TensorOpLowerToKernel.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORATensor/IR


  DEPENDS
  # ADORATensorPassIncGen
  ADORATensorUtility
  
  LINK_LIBS PUBLIC
  ADORATensorOps
  MLIRTensorDialect
  ADORATensorOpGemmLower
  MLIRADORATransformsUtility

  # GeneralCDFGGen
  # MLIRADORALoopTransforms

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

add_subdirectory(Gemm)