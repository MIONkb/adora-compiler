get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  # Core
  MLIRIR
  MLIRTransforms
  MLIROptLib
  
  # Dialects
  MLIRTosaDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRMLProgramDialect
  MLIRTensorDialect

 
  # Transforms
  MLIRFuncTransforms
  MLIRLinalgTransforms
  MLIRAffineTransforms
  MLIRReconcileUnrealizedCasts
  MLIRMemRefTransforms
  
  # Conversion
  MLIRLinalgTransforms
  MLIRLinalgToStandard

  #ADORA
  MLIRADORAOps
  # MLIRADORALowerings
  # MLIRADORAMisc
  # MLIRADORATransforms
  # GeneralCDFGGen

  #ADORATensor
  ADORATensorOps
  ADORATensorTransforms
)

set(SOURCES
  tensor-opt.cpp
)

add_llvm_executable(tensor-opt ${SOURCES})

target_link_libraries(tensor-opt PRIVATE ${LIBS})
llvm_update_compile_flags(tensor-opt)

mlir_check_link_libraries(tensor-opt)

install(TARGETS tensor-opt)