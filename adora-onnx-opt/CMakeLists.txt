set(_SAVE_CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
set(_SAVE_CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}")

set(CMAKE_PREFIX_PATH "")
set(CMAKE_MODULE_PATH "")

set(ONNX_LLVM_CMAKE_DIR
    "${THIRDPARTY_ONNX_PRJ_DIR}/third_party/llvm-project-onnx/build/lib/cmake/llvm")
set(ONNX_MLIR_CMAKE_DIR
    "${THIRDPARTY_ONNX_PRJ_DIR}/third_party/llvm-project-onnx/build/lib/cmake/mlir")

list(APPEND CMAKE_PREFIX_PATH "${ONNX_LLVM_CMAKE_DIR}" "${ONNX_MLIR_CMAKE_DIR}")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${ONNX_MLIR_CMAKE_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${ONNX_LLVM_CMAKE_DIR}")

set(CMAKE_BINARY_DIR_FOR_ONNX ${CMAKE_BINARY_DIR}/adora-onnx-opt)
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR_FOR_ONNX}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR_FOR_ONNX}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR_FOR_ONNX})

include(TableGen)
include(AddLLVM)
# include(AddMLIR)
include(HandleLLVMOptions)
# include(MLIRDetectPythonEnv)

set(ONNX_LLVM_INCLUDE_DIRS "${ONNX_LLVM_BUILD_DIR}/include")
set(ONNX_LLVM_LIBRARY_DIRS "${ONNX_LLVM_INSTALL_DIR}/lib")
message(ONNX_LLVM_INCLUDE_DIRS: ${ONNX_LLVM_INCLUDE_DIRS})
message(ONNX_LLVM_LIBRARY_DIRS: ${ONNX_LLVM_LIBRARY_DIRS})

###### compile ADORATensor Dialect
add_subdirectory(lib/Dialect/ADORATensor/IR)

# To solve error: exception handling disabled, use '-fexceptions' to enable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPDLOG_NO_EXCEPTIONS ")

set(LIBS
  # Core
  MLIRIR
  MLIRTransforms
  MLIROptLib
  
  # Dialects
  # MLIRTosaDialect
  # MLIRLinalgDialect
  # MLIRMemRefDialect
  # MLIRAffineDialect
  # MLIRArithDialect
  # MLIRMathDialect
  # MLIRFuncDialect
  # MLIRSCFDialect
  # MLIRMLProgramDialect
  # MLIRTensorDialect

 
  # Transforms
  # MLIRFuncTransforms
  # MLIRLinalgTransforms
  # MLIRAffineTransforms
  # MLIRReconcileUnrealizedCasts
  # MLIRMemRefTransforms
  
  # Conversion
  # MLIRLinalgTransforms
  # MLIRLinalgToStandard

  #ADORA
  # MLIRADORAOps
  # MLIRADORAKernelExtract
  # MLIRADORALowerings
  # MLIRADORAMisc
  # MLIRADORATransforms
  # GeneralCDFGGen

  #ADORATensor
  ADORATensorOpsForONNX
  # ADORATensorTransforms
  # ADORATensorLower
)

set(SOURCES
  adora-onnx-opt.cpp
)

add_llvm_executable(adora-onnx-opt ${SOURCES})

target_link_libraries(adora-onnx-opt PRIVATE ${LIBS})
set_target_properties(adora-onnx-opt PROPERTIES INCLUDE_DIRECTORIES "")
target_include_directories(adora-onnx-opt
  PRIVATE
    "${ONNX_LLVM_INCLUDE_DIRS}"
    "${THIRDPARTY_ONNX_PRJ_DIR}" 
    "${THIRDPARTY_ONNX_PRJ_DIR}/build" 
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/build/include"
)

llvm_update_compile_flags(adora-onnx-opt)

mlir_check_link_libraries(adora-onnx-opt)

install(TARGETS adora-onnx-opt)


set(CMAKE_PREFIX_PATH "${_SAVE_CMAKE_PREFIX_PATH}")
set(CMAKE_MODULE_PATH "${_SAVE_CMAKE_MODULE_PATH}")