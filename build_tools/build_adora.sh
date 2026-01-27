#!/bin/bash
# 获取当前脚本所在目录的上一级，即项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 使用 env.sh 中定义的变量，如果没定义则报错
if [ -z "$LLVM_PROJ_BUILD" ]; then
  echo "Error: LLVM_PROJ_BUILD is not set. Please source env.sh first."
  exit 1
fi

LLVM_INSTALL_DIR="${LLVM_PROJ_BUILD}"
BUILD_DIR="${PROJECT_ROOT}/build"

# 进入项目根目录
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
# 编译并运行测试
ninja -j 32 install check-adora