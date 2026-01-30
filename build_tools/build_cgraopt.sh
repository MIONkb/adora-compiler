#!/bin/bash
# Get the parent directory of this script, i.e. the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use variables defined in env.sh; error out if not defined
if [ -z "$LLVM_PROJ_BUILD" ]; then
  echo "Error: LLVM_PROJ_BUILD is not set. Please source env.sh first."
  exit 1
fi

LLVM_INSTALL_DIR="${LLVM_PROJ_BUILD}"
BUILD_DIR="${PROJECT_ROOT}/build"

# Enter the project root directory
cd "${PROJECT_ROOT}"

if [ ! -d "$BUILD_DIR" ]; then
  mkdir -p build && cd build

  cmake -GNinja \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=. \
    -DMLIR_DIR=$LLVM_INSTALL_DIR/lib/cmake/mlir \
    -DLLVM_BUILD_DIR=$LLVM_INSTALL_DIR \
    -DLLVM_INSTALL_DIR=$LLVM_INSTALL_DIR \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    .. 

else
    echo "Using existing build directory for incremental build..."
    cd "$BUILD_DIR"
fi
# Build and run tests
ninja -j 32 install check-adora
