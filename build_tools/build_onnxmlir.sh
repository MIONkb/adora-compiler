#!/bin/bash
# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "$LLVM_PROJ_BUILD" ]; then
  echo "Error: LLVM_PROJ_BUILD is not set. Please source env.sh first."
  exit 1
fi

# Path definitions
ONNX_PROJ_DIR="${PROJECT_ROOT}/frontend/adora-onnx-mlir"
MLIR_DIR="${LLVM_PROJ_BUILD}/lib/cmake/mlir"
ADORA_INSTALL_DIR="${PROJECT_ROOT}/build"
BUILD_DIR="${ONNX_PROJ_DIR}/build"

echo "Building Adora ONNX Frontend..."
echo "Source: $ONNX_PROJ_DIR"
echo "Linking against Adora Backend: $ADORA_INSTALL_DIR"


if [ ! -d "$BUILD_DIR" ]; then
  mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

  cmake -G Ninja \
          -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
          -DCMAKE_BUILD_TYPE=Debug \
          -DLLVM_ENABLE_ASSERTIONS=ON \
          -DMLIR_DIR=${MLIR_DIR} \
          -DADORA_INSTALL_DIR=${ADORA_INSTALL_DIR} \
          -DCMAKE_INSTALL_PREFIX=$ONNX_PROJ_DIR/build  \
          -DBUILD_SHARED_LIBS=OFF \
          ..
else
  cd "$BUILD_DIR"
fi

ninja -j 16 install
