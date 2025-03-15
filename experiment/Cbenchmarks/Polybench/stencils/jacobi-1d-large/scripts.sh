
clear && \
cgeist \
    -O2 \
    -lm -lgcc \
    -Dsize_t=int -Dwint_t=int -DROCKET_TARGET -D_riscv \
    -DLARGE_DATASET \
    --import-all-index \
    -I/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/stencils/jacobi-1d-large \
    -I/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/utilities \
    -I/home/jhlou/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/include/machine/ \
    -I/home/jhlou/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/include/ \
    -I/home/jhlou/chipyard/.conda-env/riscv-tools/sysroot/usr/include/linux/ \
    -I/home/jhlou/chipyard/.conda-env/riscv-tools/lib/gcc/riscv64-unknown-elf/12.2.0/include \
    -I/home/jhlou/chipyard/generators/fdra/software/tests/riscv-tests \
    -I/home/jhlou/chipyard/generators/fdra/software/tests/riscv-tests/env \
    -I/home/jhlou/chipyard/generators/fdra/software/tests \
    -I/home/jhlou/chipyard/generators/fdra/software/tests/riscv-tests/benchmarks/common \
   /home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/stencils/jacobi-1d-large/jacobi-1d.c \
    -S -o jacobi-1d.mlir

mlir-opt\
 --allow-unregistered-dialect      \
 --affine-loop-normalize \
 --affine-simplify-structures   \
 --normalize-memrefs \
 --force-specialization \
 --bufferization-bufferize \
 jacobi-1d.mlir \
 -o jacobi-1d_normalized.mlir


cgra-opt \
    --canonicalize \
    -reconcile-unrealized-casts \
    --arith-expand --memref-expand \
    -cse \
    --affine-loop-fusion \
    --ADORA-extract-kernel-to-function="kernel-gen-dir=$PWD" \
    jacobi-1d_normalized.mlir -o jacobi-1d_normalized_host.mlir


    
cgra-opt \
    --ADORA-extract-affine-for-to-kernel \
    --ADORA-adjust-kernel-mem-footprint="cachesize=64 singlearraysize=4 disable-remainder-block explicit-datablock" \
    jacobi_1d_kernel_0.mlir

cgra-opt \
    --ADORA-kernel-dfg-gen \
    /home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/linear-algebra/kernels/bicg_small/1_kernels_opt/bicg_kernel_opt.mlir


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