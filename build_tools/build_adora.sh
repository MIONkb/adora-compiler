#!/bin/bash
### set LLVM_BUILD_DIR to your own llvm path
LLVM_BUILD_DIR="/home/share/llvm-project-Polygeist/build/"
LLVM_INSTALL_DIR="/home/share/llvm-project-Polygeist/build/"
THIRDPARTY_ONNX_PRJ_DIR=/home/jhlou/CGRVOPT/onnx-mlir

mkdir build && cd build
# ......................................................................
cmake -GNinja \
  ..\
  "-B." \
  -DCMAKE_INSTALL_PREFIX=. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit \
  -DMLIR_DIR=$LLVM_INSTALL_DIR/lib/cmake/mlir \
  -DLLVM_BUILD_DIR=$LLVM_BUILD_DIR \
  -DLLVM_INSTALL_DIR=$LLVM_INSTALL_DIR \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DTHIRDPARTY_ONNX_PRJ_DIR=$THIRDPARTY_ONNX_PRJ_DIR


  # -DADORA_ENABLE_ONNX_TENSOR_OPT=ON

# cmake --build . --target cgra-opt cgra-mapper
ninja -j 32 install