add_mlir_dialect_library(ADORATensorOps
  Ops/OpHelper.cpp
  Ops/Matmul.cpp
  ADORATensorDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/RAAA/Dialect/ADORATensor/IR

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