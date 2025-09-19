cmake_minimum_required(VERSION 3.13.4)
project(cgra-opt LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(RAAA_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(MLIRDetectPythonEnv)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

set(LLVM_INCLUDE_DIRS "${LLVM_BUILD_DIR}/include")
set(LLVM_LIBRARY_DIRS "${LLVM_INSTALL_DIR}/lib")
message(LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS})
message(LLVM_LIBRARY_DIRS: ${LLVM_LIBRARY_DIRS})

link_directories(${LLVM_LIBRARY_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/mapper/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/lib/DFG/inc)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# set(THIRDPARTY_ONNX_PRJ_DIR "/home/jhlou/CGRVOPT/onnx-mlir"
#     CACHE PATH "Path to onnx-mlir project (source/build root)")
option(THIRDPARTY_ONNX_PRJ_DIR "onnx-mlir project path" third_party/onnx-mlir)
include_directories(${THIRDPARTY_ONNX_PRJ_DIR}/build)
include_directories(${THIRDPARTY_ONNX_PRJ_DIR})

add_subdirectory(include/ADORA)
add_subdirectory(lib)
add_subdirectory(mapper)
# add_subdirectory(test)
add_subdirectory(tools)



## third party
## onnx
# option(ADORA_ENABLE_ONNX_TENSOR_OPT "Enable ONNX→ADORA tensor pipeline" OFF)
# if(ADORA_ENABLE_ONNX_TENSOR_OPT)
#   set(THIRDPARTY_ONNX_PRJ_DIR "/home/jhlou/CGRVOPT/onnx-mlir"
#     CACHE PATH "Path to onnx-mlir project (source/build root)")
#   add_subdirectory(adora-onnx-opt)
# endif()