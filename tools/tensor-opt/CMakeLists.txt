get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# To solve error: exception handling disabled, use '-fexceptions' to enable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPDLOG_NO_EXCEPTIONS ")

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
  MLIRADORAKernelExtract
  # MLIRADORALowerings
  # MLIRADORAMisc
  # MLIRADORATransforms
  # GeneralCDFGGen

  #ADORATensor
  ADORATensorOps
  ADORATensorTransforms
  ADORATensorLower
)

set(SOURCES
  tensor-opt.cpp
)

add_llvm_executable(tensor-opt ${SOURCES})

target_link_libraries(tensor-opt PRIVATE ${LIBS})
llvm_update_compile_flags(tensor-opt)

mlir_check_link_libraries(tensor-opt)

install(TARGETS tensor-opt)