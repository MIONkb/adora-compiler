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
    linalg_conv2d.mlir \
    -o affine.mlir

llc -O3 /home/jhlou/CGRVOPT/cgra-opt/experiment/AILayers/conv2d/forward_opt.ll \
  -I /home/jhlou/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/include \
  -march=riscv64 -mtriple=riscv64-unknown-elf-gnu -mcpu=rocket-rv64 \
  -mattr=+c,+d,+relax,+m  \
  --relocation-model=pic \
  -float-abi=hard \
  -code-model=small \
  -o /home/jhlou/CGRVOPT/cgra-opt/experiment/AILayers/conv2d/4_asms/forward.s