mlir-opt \
    --eliminate-empty-tensors \
    --empty-tensor-to-alloc-tensor \
     --arith-bufferize \
    --tensor-bufferize \
    --func-bufferize \
    --linalg-bufferize  \
    --finalizing-bufferize \
    --convert-bufferization-to-memref  \
    --convert-linalg-to-affine-loops \
    --affine-scalrep \
    --affine-simplify-structures  \
    --canonicalize \
    linalg_linear.mlir \
    -o affine.mlir

cgra-opt --adora-convert-linalg-to-systolic-gemm linalg_linear_elide.mlir 

source scripts/env.sh
cgra-mapper \
    --adg="${CGRA_ADG_PATH}/cgra_adg.json" \
    --op-file="${CGRA_OP_FILE_PATH}/operations.json" \
    --output="/home/jhlou/CGRVOPT/cgra-opt/experiment/AILayers/linear/linear.py" \
    --output-type="pytest" \
    "/home/jhlou/CGRVOPT/cgra-opt/experiment/AILayers/linear/IR/0_kernels/gemm.mlir" 