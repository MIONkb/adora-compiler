#!/bin/bash
# This script was modified from:
# https://raw.githubusercontent.com/tensorflow/mlir-hlo/master/build_tools/build_mlir.sh

set -e

### set LLVM_SRC_DIR to your own path
LLVM_SRC_DIR=/home/jhlou/CGRVOPT/llvm-project-Polygeist

build_dir=$LLVM_SRC_DIR/build
install_dir=$LLVM_SRC_DIR/build

echo "Using LLVM source dir: $LLVM_SRC_DIR"

# Setup directories.
echo "Building MLIR in $build_dir"
mkdir -p "$build_dir"
echo "Creating directory to install: $install_dir"
mkdir -p "$install_dir"

echo "Beginning build (commands will echo)"
set -x

cmake -GNinja \
  "-H$LLVM_SRC_DIR/llvm" \
  "-B$build_dir" \
  -DCMAKE_INSTALL_PREFIX=$install_dir  \
  -DLLVM_INSTALL_UTILS=ON   \
  -DLLVM_ENABLE_PROJECTS="mlir;clang"   \
  -DLLVM_TARGETS_TO_BUILD="host;RISCV"   \
  -DLLVM_INCLUDE_TOOLS=ON   \
  -DLLVM_BUILD_TOOLS=ON   \
  -DLLVM_INCLUDE_TESTS=ON   \
  -DMLIR_INCLUDE_TESTS=ON   \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DLLVM_ENABLE_LLD=OFF

 # TODO check what these options do :
  # -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  #  -DLLVM_ENABLE_LLD=ON   \
 # -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_OCAMLDOC=OFF -DLLVM_ENABLE_BINDINGS=OFF 

cmake --build "$build_dir" --target opt mlir-opt mlir-translate mlir-cpu-runner install
ninja -j 16 install 