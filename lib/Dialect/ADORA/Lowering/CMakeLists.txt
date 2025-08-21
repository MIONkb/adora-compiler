add_mlir_conversion_library(MLIRADORALowerings
  KernelCallToLLVM.cpp
  MathRewrite.cpp
  ADORAToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/RAAA/Dialect/ADORA/Lowering
  ${PROJ_INCLUDE_DIR}/RAAA/Dialect/ADORA/IR

  DEPENDS
  ADORALowerPassIncGen
  MLIRADORAOps

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRAnalysis
  MLIRFuncDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRTransformUtils
  )
