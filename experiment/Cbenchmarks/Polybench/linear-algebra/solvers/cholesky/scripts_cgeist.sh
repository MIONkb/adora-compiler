clear && \
cgeist \
    -O2 \
    -lm -lgcc \
    -Dsize_t=int -Dwint_t=int -DROCKET_TARGET -D_riscv -D__GNUC__\
    -DLARGE_DATASET \
    -I/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/linear-algebra/solvers/cholesky \
    -I/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/utilities \
    -I/home/jhlou/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/include/machine/ \
    -I/home/jhlou/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/include/ \
    -I/home/jhlou/chipyard/.conda-env/riscv-tools/sysroot/usr/include/linux/ \
    -I/home/jhlou/chipyard/.conda-env/riscv-tools/lib/gcc/riscv64-unknown-elf/12.2.0/include \
    -I/home/jhlou/chipyard/generators/fdra/software/tests/riscv-tests \
    -I/home/jhlou/chipyard/generators/fdra/software/tests/riscv-tests/env \
    -I/home/jhlou/chipyard/generators/fdra/software/tests \
    -I/home/jhlou/chipyard/generators/fdra/software/tests/riscv-tests/benchmarks/common \
   /home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/linear-algebra/solvers/cholesky/cholesky.c \
    -S -o cholesky.mlir


## options:
# -emit-llvm

# clang \
#     -I /home/jhlou/CGRVOPT/Benchmarks/PolyBenchC-4.2.1-master/datamining/correlation \
#     -I /home/jhlou/CGRVOPT/Benchmarks/PolyBenchC-4.2.1-master/utilities \
#     /home/jhlou/CGRVOPT/Benchmarks/PolyBenchC-4.2.1-master/datamining/correlation/correlation.c \
#     -DDATA_TYPE_IS_FLOAT \
#     -emit-llvm -S \
#     -o correlation.ll

# riscv64-unknown-elf-gcc \
#         -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany \
#         -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf \
#         -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc12 \
#         -lm -lgcc \
#         -I/home/tianyi/chipyard/generators/fdra/software/tests/riscv-tests \
#         -I/home/tianyi/chipyard/generators/fdra/software/tests/riscv-tests/env \
#         -I/home/tianyi/chipyard/generators/fdra/software/tests/ \
#         -I/home/tianyi/chipyard/generators/fdra/software/tests/riscv-tests/benchmarks/common \
#         -DID_STRING=  -nostdlib -nostartfiles -static \
#         -DBAREMETAL=1 \
#         "$file" \
#         -S -o "$tarfolder/$filename.s"