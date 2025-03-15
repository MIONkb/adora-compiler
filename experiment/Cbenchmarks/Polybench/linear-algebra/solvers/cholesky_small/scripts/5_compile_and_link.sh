#!/bin/bash

# 指定文件夹路径
rootfolder=$(pwd)
asmfolder="$rootfolder/4_asms"
tarfolder="/home/jhlou/chipyard/generators/fdra/software/tests/bareMetalC"
tempfolder="$rootfolder/tempfiles"
kernel_basename="forward_kernel"
top_call_name="forward"
main_file="cholesky_main.c"

if [ ! -d "$tarfolder" ]; then
  mkdir -p "$tarfolder"
  echo mkdir -p "$tarfolder"
fi
if [ ! -d "$tempfolder" ]; then
  mkdir -p "$tempfolder"
  echo mkdir -p "$tempfolder"
fi
# cd $tempfolder
# if [[ "$(pwd)" == "$tempfolder" ]]; then
#   find "$tempfolder" -name "*.dot" -type f -delete
#   find "$tempfolder" -name "*.text" -type f -delete
#   cd -
# fi


# asm to object
# asm_files=$(find "$asmfolder" -name "*.s" -type f)
# cnt=0

# sed -i -E 's/^(\s*)(\.file|\.loc)/\1#&/' $asmfolder/"$top_call_name.s"

# for file in ${asm_files[@]}; do
#     # 检查文件是否为普通文件
#     filename=$(basename "$file" .s)
#     echo "$filename"
#     if [[ -f "$file" ]]; then
#       echo $cnt
#       ((cnt++))

#       # gcc .c to .asm
#       echo  riscv64-unknown-elf-as -march=rv64gc \
#           $file -o $asmfolder/$filename.o 

#       riscv64-unknown-elf-as -march=rv64gc \
#           $file -o $asmfolder/$filename.o 
#     fi
# done
# cnt=0

# link objects
object_files=$(find "$asmfolder" -name "*.s" -type f)

echo \
riscv64-unknown-elf-gcc \
  -DROCKET_TARGET  \
  -DSMALL_DATASET \
 -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany \
  -O2 -ffast-math -fno-common -fno-builtin-printf \
 -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc12 \
 -lm  \
 -I/home/jhlou/chipyard/generators/fdra/software/tests//riscv-tests/benchmarks/common \
 -I/home/jhlou/chipyard/generators/fdra/software/tests/riscv-tests \
 -I/home/jhlou/chipyard/generators/fdra/software/tests//riscv-tests/env \
 -I/home/jhlou/chipyard/generators/fdra/software/tests/ \
  -I/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/linear-algebra/solvers/cholesky \
  -I/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/utilities \
 -T/home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/my_test.ld \
 -DID_STRING= -nostartfiles -static  \
 -DBAREMETAL=1 -e _start -g  \
 -o /home/jhlou/chipyard/generators/fdra/software/tests/build/bareMetalC/cholesky-small-noprint-fence-baremetal \
 $rootfolder/$main_file \
 $object_files \
 /home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/syscalls.c \
  /home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/CRunnerUtils.cpp \
  /home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/tiny-malloc.c 

riscv64-unknown-elf-gcc \
  -DROCKET_TARGET  \
  -DSMALL_DATASET \
 -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany \
  -O2 -ffast-math -fno-common -fno-builtin-printf \
 -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc12 \
 -lm  \
 -I/home/jhlou/chipyard/generators/fdra/software/tests//riscv-tests/benchmarks/common \
 -I/home/jhlou/chipyard/generators/fdra/software/tests/riscv-tests \
 -I/home/jhlou/chipyard/generators/fdra/software/tests//riscv-tests/env \
 -I/home/jhlou/chipyard/generators/fdra/software/tests/ \
  -I/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/linear-algebra/solvers/cholesky \
  -I/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/utilities \
 -T/home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/my_test.ld \
 -DID_STRING= -nostartfiles -static  \
 -DBAREMETAL=1 -e _start -g  \
 -o /home/jhlou/chipyard/generators/fdra/software/tests/build/bareMetalC/cholesky-small-noprint-fence-baremetal \
 $rootfolder/$main_file \
 $object_files \
 /home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/syscalls.c \
  /home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/CRunnerUtils.cpp \
  /home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/tiny-malloc.c 


# -DDEFINE_MALLOC -lgcc
#  /home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/tiny-malloc.c \
#  /home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/CRunnerUtils.cpp 
#  /home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/utilities/rocket_polybench.c
#  /home/jhlou/chipyard/generators/fdra/software/tests//riscv-tests/benchmarks/common/syscalls.c \
#  /home/jhlou/chipyard/generators/fdra/software/tests/gemm/crt.S \
#  /home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/CRunnerUtils.cpp
#  /home/jhlou/chipyard/generators/fdra/software/tests/UtilSrc/tiny-malloc.c 



#  "$asmfolder"/forward_kernel_1.s \
#  "$asmfolder"/forward_kernel_0.s \
#  "$asmfolder"/syscalls.s \
#  "$asmfolder"/crt.s \
#  "$asmfolder"/forward_kernel_2.s\
#  "$asmfolder"/forward.s \