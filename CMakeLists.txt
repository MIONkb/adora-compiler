cmake_minimum_required(VERSION 3.20.0)
project(ADORA VERSION 0.1.0 LANGUAGES CXX C)

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
include(CMakePackageConfigHelpers)

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
list(APPEND CMAKE_PREFIX_PATH "${THIRDPARTY_ONNX_PRJ_DIR}/build")
message(THIRDPARTY_ONNX_PRJ_DIR: ${THIRDPARTY_ONNX_PRJ_DIR})
# find_package(onnx-mlir CONFIG QUIET)
# if(NOT onnx-mlir_FOUND)
#   find_package(ONNX CONFIG REQUIRED)
# endif()

include_directories(${THIRDPARTY_ONNX_PRJ_DIR}/build)
include_directories(${THIRDPARTY_ONNX_PRJ_DIR})
# link_directories(${THIRDPARTY_ONNX_PRJ_DIR}/build/release/lib)
# find_package(onnx-mlir CONFIG QUIET)
# if(NOT onnx-mlir_FOUND)
#   find_package(ONNX_MLIR CONFIG REQUIRED)
# endif()

add_subdirectory(include/ADORA)
add_subdirectory(lib)
add_subdirectory(mapper)
# add_subdirectory(test)
add_subdirectory(tools)
# add_subdirectory(cmake)

### install .a libs
install(
  TARGETS MLIRADORAOps MLIRADORALowerings MLIRADORATransforms MLIRADORATransformsUtility MLIRADORALoopTransforms MLIRADORAKernelExtract  GeneralCDFGGen
  TARGETS ADORATensorInterface ADORATensorOps ADORATensorLower ADORATensorTransforms ADORATensorUtility ADORATensorOpGemmLower
  EXPORT ADORATargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  # RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

### generate ADORATargets.cmake
install(EXPORT ADORATargets
  NAMESPACE ADORA::
  FILE ADORATargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ADORA
)

# ADORAConfig.cmake / ADORAConfigVersion.cmake
set(ADORA_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/ADORA)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ADORAConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/ADORAConfig.cmake
  INSTALL_DESTINATION ${ADORA_INSTALL_CMAKEDIR}
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
# write_basic_package_version_file(
#   ${CMAKE_CURRENT_BINARY_DIR}/ADORAConfigVersion.cmake
#   VERSION ${PROJECT_VERSION}
#   COMPATIBILITY SameMajorVersion
# )
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/ADORAConfig.cmake
  # ${CMAKE_CURRENT_BINARY_DIR}/ADORAConfigVersion.cmake
  DESTINATION ${ADORA_INSTALL_CMAKEDIR}
)
## third party
## onnx
# option(ADORA_ENABLE_ONNX_TENSOR_OPT "Enable ONNX→ADORA tensor pipeline" OFF)
# if(ADORA_ENABLE_ONNX_TENSOR_OPT)
#   set(THIRDPARTY_ONNX_PRJ_DIR "/home/jhlou/CGRVOPT/onnx-mlir"
#     CACHE PATH "Path to onnx-mlir project (source/build root)")
#   add_subdirectory(adora-onnx-opt)
# endif()