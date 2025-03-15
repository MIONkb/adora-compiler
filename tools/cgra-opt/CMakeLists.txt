# This includes several libs that extends the linking time
# only the necessary libs are included instead
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# message(PUBLIC "${dialect_libs}")
# message(PUBLIC "${conversion_libs}")

set(LIBS
  # ${dialect_libs}
  # ${conversion_libs}

  # Core
  MLIRIR
  MLIRTransforms
  MLIROptLib
  
  # Dialects
  MLIRLLVMDialect
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
  MLIRSCFTransforms
  MLIRReconcileUnrealizedCasts
  MLIRMemRefTransforms

  # Test Passes
  MLIRLinalgTestPasses
  MLIRAffineTransformsTestPasses
  
  # Conversion
  MLIRAffineToStandard
  MLIRSCFToControlFlow
  MLIRMemRefToLLVM
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRLinalgTransforms
  MLIRLinalgToStandard

  #ADORA
  MLIRADORAMisc
  MLIRADORAOps
  MLIRADORATransforms
  MLIRADORALowerings
)

set(SOURCES
  cgra-opt.cpp
)

add_llvm_executable(cgra-opt ${SOURCES})

# llvm_update_compile_flags(cgra-opt)
target_link_libraries(cgra-opt PRIVATE ${LIBS})
llvm_update_compile_flags(cgra-opt)

mlir_check_link_libraries(cgra-opt)

install(TARGETS cgra-opt)