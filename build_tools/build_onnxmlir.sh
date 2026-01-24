#!/bin/bash
# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "$LLVM_PROJ_BUILD" ]; then
  echo "Error: LLVM_PROJ_BUILD is not set. Please source env.sh first."
  exit 1
fi

# 指向你自己目录下的 frontend 代码
ONNX_PROJ_DIR="${PROJECT_ROOT}/frontend/adora-onnx-mlir"

# 使用借用的 LLVM
MLIR_DIR="${LLVM_PROJ_BUILD}/lib/cmake/mlir"

# Adora 后端安装目录
ADORA_INSTALL_DIR="${PROJECT_ROOT}/build"

echo "Building Adora ONNX Frontend..."
echo "Source: $ONNX_PROJ_DIR"
echo "Linking against Adora Backend: $ADORA_INSTALL_DIR"

mkdir -p $ONNX_PROJ_DIR/build && cd $ONNX_PROJ_DIR/build

cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_DIR=${MLIR_DIR} \
        -DADORA_INSTALL_DIR=${ADORA_INSTALL_DIR} \
        -DCMAKE_INSTALL_PREFIX=$ONNX_PROJ_DIR/build  \
        -DBUILD_SHARED_LIBS=OFF \
        -DPython3_EXECUTABLE=/usr/bin/python3 \
        -DPython3_INCLUDE_DIR=/usr/include/python3.8 \
        -DPython3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so \
        \
        -DLLVM_ENABLE_WERROR=OFF \
        -DCMAKE_CXX_FLAGS="-Wno-error -w" \
        -DCMAKE_C_FLAGS="-Wno-error -w" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        \
        ..

ninja -j 16 install