mlir-opt\
 --allow-unregistered-dialect      \
 --affine-loop-normalize \
 --affine-simplify-structures   \
 --normalize-memrefs \
 --force-specialization \
 --bufferization-bufferize \
 correlation.mlir \
 -o correlation_normalized.mlir


cgra-opt \
    --canonicalize \
    -reconcile-unrealized-casts \
    --arith-expand --memref-expand \
    -cse \
    --affine-loop-fusion \
    --ADORA-extract-affine-for-to-kernel="function-name=kernel_correlation" \
    --ADORA-extract-kernel-to-function="kernel-gen-dir=$PWD" \
    correlation_normalized.mlir -o correlation_host.mlir

    --ADORA-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock"