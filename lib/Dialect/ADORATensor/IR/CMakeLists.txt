add_mlir_dialect_library(ADORATensorOps
  Ops/OpHelper.cpp
  Ops/MatMul.cpp
  Ops/Gemm.cpp
  Ops/Conv.cpp
  ADORATensorDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORATensor/IR

  DEPENDS

  LINK_LIBS PUBLIC
  ADORATensorInterface
  MLIRIR
  MLIRPass
  #MLIRTransforms
  #MLIRSCFDialect
  #MLIRAffineDialect
  #MLIRAffineToStandard
  )

set(LIBS
  ADORATensorOps
)