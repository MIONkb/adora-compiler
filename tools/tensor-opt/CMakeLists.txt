get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# To solve error: exception handling disabled, use '-fexceptions' to enable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPDLOG_NO_EXCEPTIONS ")

# set(ONNX_MLIR_LIB_DIR
#     "${THIRDPARTY_ONNX_PRJ_DIR}/build/Debug/lib"
#     CACHE PATH "Path to onnx-mlir libs (.a/.so)")
# set(ONNX_MLIR_INCLUDE_DIR
#     "${THIRDPARTY_ONNX_PRJ_DIR}/build/Release/include"
#     CACHE PATH "Path to onnx-mlir generated headers")

# function(_import_static name libfile)
#   if(NOT IS_ABSOLUTE "${libfile}")
#     set(libfile "${ONNX_MLIR_LIB_DIR}/${libfile}")
#   endif()
#   if(NOT EXISTS "${libfile}")
#     message(FATAL_ERROR "Missing ${libfile} for target ${name}")
#   endif()
#   add_library(${name} STATIC IMPORTED)
#   set_target_properties(${name} PROPERTIES
#     IMPORTED_LOCATION "${libfile}"
#     # INTERFACE_INCLUDE_DIRECTORIES "${ONNX_MLIR_INCLUDE_DIR}"
#   )
# endfunction()

# _import_static(onnx_mlir::OMONNXOps               libOMONNXOps.a)

# _import_static(onnx_mlir::OMSupport               libOMSupport.a)
# _import_static(onnx_mlir::OMONNXConversionCommon  libOMONNXConversionCommon.a)
# _import_static(onnx_mlir::OMONNXRewrite           libOMONNXRewrite.a)
# _import_static(onnx_mlir::OMONNXElementsAttr      libOMONNXElementsAttr.a)
# _import_static(onnx_mlir::OMMlirUtilities         libOMMlirUtilities.a)
# _import_static(onnx_mlir::OMMlirDialects          libOMMlirDialects.a)
# _import_static(onnx_mlir::OMDiagnostic            libOMDiagnostic.a)
# _import_static(onnx_mlir::OMCompilerOptions       libOMCompilerOptions.a)
# _import_static(onnx_mlir::OMSmallFPConversion     libOMSmallFPConversion.a)
# _import_static(onnx_mlir::OMAccelerator           libOMAccelerator.a)
# _import_static(onnx_mlir::OMShapeHelperOpInterface     libOMShapeHelperOpInterface.a)

set(LIBS
  # Core
  MLIRIR
  MLIRTransforms
  MLIROptLib
  # MLIRParser
  # MLIRSupport
  # MLIRAsmParser
  
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
  MLIRADORAKernelOp
  # MLIRADORALowerings
  # MLIRADORAMisc
  # MLIRADORATransforms
  GeneralCDFGGen

  #ADORATensor
  ADORATensorOps
  ADORATensorTransforms
  ADORATensorLower

  # ONNX-MLIR
  # onnx_mlir::OMONNXOps
  # onnx_mlir::OMSupport 
  # onnx_mlir::OMONNXConversionCommon
  # onnx_mlir::OMONNXRewrite
  # onnx_mlir::OMONNXElementsAttr
  # onnx_mlir::OMMlirUtilities 
  # onnx_mlir::OMMlirDialects 
  # onnx_mlir::OMDiagnostic 
  # # onnx_mlir::OMCompilerOptions
  # onnx_mlir::OMSmallFPConversion
  # onnx_mlir::OMAccelerator
  # MLIRShapeDialect
  # onnx_mlir::OMShapeHelperOpInterface
)

set(SOURCES
  tensor-opt.cpp
)

add_llvm_executable(tensor-opt ${SOURCES})


target_link_libraries(tensor-opt PRIVATE ${LIBS})
llvm_update_compile_flags(tensor-opt)

mlir_check_link_libraries(tensor-opt)

install(TARGETS tensor-opt)