mlir-opt\
 --allow-unregistered-dialect      \
 --affine-loop-normalize \
 --affine-simplify-structures   \
 --normalize-memrefs \
 --force-specialization \
 --bufferization-bufferize \
 cholesky.mlir \
 -o cholesky_normalized.mlir


cgra-opt \
    --canonicalize \
    -reconcile-unrealized-casts \
    --arith-expand --memref-expand \
    -cse \
    --affine-loop-fusion \
    --ADORA-extract-kernel-to-function="kernel-gen-dir=$PWD" \
    cholesky_normalized.mlir -o cholesky_host.mlir

    --ADORA-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock"


opt -O3 --disable-builtin=memset --mtriple=riscv64 --mcpu=rocket-rv64 cholesky.ll -S -o cholesky_riscv.ll

clang \
 --target=riscv64 \
 -mcpu=rocket-rv64 \
 cholesky_opt.ll -o cholesky.s

  --mtriple=riscv64-unknown-elf-gnu \


llc \
  -O3 -I /home/jhlou/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/include\
  -march=riscv64 -mtriple=riscv64-unknown-elf-gnu -mcpu=rocket-rv64 \
  -mattr=+c,+d,+relax,+m  \
  --relocation-model=pic \
  -float-abi=hard \
  -code-model=small \
  cholesky_opt.ll -o cholesky.s