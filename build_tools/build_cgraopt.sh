#!/bin/bash
### set LLVM_BUILD_DIR to your own llvm path
LLVM_BUILD_DIR=/home/jhlou/CGRVOPT/llvm-project-Polygeist/build
LLVM_INSTALL_DIR=/home/jhlou/CGRVOPT/llvm-project-Polygeist/build

# ......................................................................
cmake -GNinja \
  ..\
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit \
  -DMLIR_DIR=$LLVM_INSTALL_DIR/lib/cmake/mlir \
  -DLLVM_BUILD_DIR=$LLVM_BUILD_DIR \
  -DLLVM_INSTALL_DIR=$LLVM_INSTALL_DIR \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON

# cmake --build . --target cgra-opt cgra-mapper
ninja -j 8 
