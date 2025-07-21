conda activate ~/chipyard/.conda-env/

riscv64-unknown-elf-gcc \
 -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany \
  -O0 -ffast-math -fno-common -fno-builtin-printf \
 -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc12 \
  -lm -lgcc \
 -I/home/jhlou/chipyard/generators/fdra/software/tests//riscv-tests/benchmarks/common \
 -I/home/jhlou/chipyard/generators/fdra/software/tests/riscv-tests \
 -I/home/jhlou/chipyard/generators/fdra/software/tests//riscv-tests/env \
 -I/home/jhlou/chipyard/generators/fdra/software/tests/ \
 -T/home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/my_test.ld\
 -DID_STRING= -nostartfiles -static  \
 -DBAREMETAL=1 -DDEFINE_MALLOC -e _start -g  \
 -o /home/jhlou/chipyard/generators/fdra/software/tests/build/bareMetalC/maintest5-baremetal \
 /home/jhlou/CGRVOPT/cgra-opt/experiment/maintest/main.c \
 /home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/syscalls.c \
 /home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/crt.s \
  /home/jhlou/chipyard/generators/fdra/software/tests/riscv-tests/debug/programs/tiny-malloc.c