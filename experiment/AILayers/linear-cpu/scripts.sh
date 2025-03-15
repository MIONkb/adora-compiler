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

cgra-opt  --arith-expand --memref-expand -reconcile-unrealized-casts \
 -cse --affine-loop-fusion --canonicalize \
 affine.mlir -o 2_host.mlir 

mlir-opt -promote-buffers-to-stack --arith-expand --memref-expand  \
 -normalize-memrefs --expand-strided-metadata  -lower-affine \
 --scf-for-loop-canonicalization -convert-scf-to-cf \
 --convert-math-to-llvm --convert-math-to-libm \
 --convert-arith-to-llvm \
  -normalize-memrefs  \
   --import-constants-with-refs \
 --finalize-memref-to-llvm="use-opaque-pointers" \
  --import-constants-with-refs \
   --memref-expand \
 --finalize-memref-to-llvm="use-opaque-pointers" \
 -convert-func-to-llvm=use-bare-ptr-memref-call-conv \
 -convert-func-to-llvm=use-bare-ptr-memref-call-conv \
 --finalize-memref-to-llvm="use-opaque-pointers" \
 --cse --canonicalize \
 --reconcile-unrealized-casts \
 $rootfolder/2_host.mlir -o $rootfolder/"3_${func_name}_llvm.mlir" \
 --mlir-print-ir-after-all 2>&1 | cat > "3_intermediate_${func_name}_llvm.mlir"

llc -O3 /home/jhlou/CGRVOPT/cgra-opt/experiment/AILayers/linear-cpu/forward.bc \
  -I /home/jhlou/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/include \
  -march=riscv64 -mtriple=riscv64-unknown-elf-gnu -mcpu=rocket-rv64 \
  -mattr=+c,+d,+relax,+m  \
  --relocation-model=pic \
  -float-abi=hard \
  -code-model=small \
  -o /home/jhlou/CGRVOPT/cgra-opt/experiment/AILayers/linear-cpu/4_asms/forward.s