mlir-opt \
    --pass-pipeline="builtin.module(func.func(tosa-to-arith, tosa-to-tensor, tosa-to-linalg))" \
    tosa_softmax.mlir -o linalg_softmax.mlir



mlir-opt \
    --canonicalize \
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
    linalg_softmax.mlir \
    -o affine.mlir \
    -mlir-print-ir-after-all 2>&1 | cat > intermediate_affine.mlir