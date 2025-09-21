
ONNX_PROJ_DIR=/home/jhlou/CGRVOPT/onnx-mlir
MLIR_DIR=$ONNX_PROJ_DIR/third_party/llvm-project-onnx/build/lib/cmake/mlir
MLIR_DIR=$ONNX_PROJ_DIR/third_party/llvm-project-onnx/dynamicbuild/lib/cmake/mlir

mkdir $ONNX_PROJ_DIR/build && cd $ONNX_PROJ_DIR/build
cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_DIR=${MLIR_DIR} \
        -DCMAKE_INSTALL_PREFIX=$ONNX_PROJ_DIR/build  \
        -DBUILD_SHARED_LIBS=OFF \
        ..

                # -DBUILD_SHARED_LIBS=ON \

