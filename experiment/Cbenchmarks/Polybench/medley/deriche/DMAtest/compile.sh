#!/bin/bash
riscv64-unknown-elf-gcc \
  -DROCKET_TARGET -DMINI_DATASET -Wl,--wrap,malloc \
  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany \
  -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns \
  -march=rv64gc -Wa,-march=rv64gc12 -lm \
  -I/home/jhlou/chipyard/generators/fdra/software/tests//riscv-tests/benchmarks/common \
  -I/home/jhlou/chipyard/generators/fdra/software/tests/riscv-tests \
  -T/home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/my_test.ld \
  -DID_STRING= -nostartfiles -static \
  -DDEFINE_MALLOC -DBAREMETAL=1 \
  -e _start -g \
  -o DMAtest-baremetal \
  DMAtest_main.c \
  crt.s \
  syscalls.c

  
  DMAtest_main_int.c \

cp DMAtest-baremetal /home/jhlou/chipyard/generators/fdra/software/tests/build/bareMetalC/

set +x