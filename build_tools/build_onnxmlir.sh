

ONNX_PROJ_DIR="/home/share/onnx-mlir"
MLIR_DIR="/home/jhlou/CGRVOPT/llvm-project-onnx/build/lib/cmake/mlir"
ADORA_INSTALL_DIR=/home/jhlou/CGRVOPT/cgra-opt/build

mkdir $ONNX_PROJ_DIR/build && cd $ONNX_PROJ_DIR/build
cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_DIR=${MLIR_DIR} \
        -DADORA_INSTALL_DIR=${ADORA_INSTALL_DIR} \
        -DCMAKE_INSTALL_PREFIX=$ONNX_PROJ_DIR/build  \
        -DBUILD_SHARED_LIBS=OFF \
        ..

                # -DBUILD_SHARED_LIBS=ON \
ninja -j 16 install 