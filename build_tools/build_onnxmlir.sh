
ONNX_PROJ_DIR=/home/jhlou/CGRVOPT/onnx-mlir
MLIR_DIR=$ONNX_PROJ_DIR/third_party/llvm-project-onnx/build/lib/cmake/mlir

mkdir $ONNX_PROJ_DIR/build && cd $ONNX_PROJ_DIR/build
cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_DIR=${MLIR_DIR} \
        -DCMAKE_INSTALL_PREFIX=$ONNX_PROJ_DIR/build  \
        ..