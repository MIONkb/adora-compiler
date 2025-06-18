#!/bin/bash

# 指定文件夹路径
rootfolder=$(pwd)
IRfolder="IR"
asmfolder="$rootfolder/$IRfolder/4_asms"
tarfolder="$CHIPYARD_DIR/generators/fdra/software/tests/bareMetalC"
tempfolder="$rootfolder/$IRfolder/tempfiles"
main_file="3mm_main.c"
baremetal_file_name="3mm_mini"

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
if [ -z "$CHIPYARD_DIR" ]; then
  echo "Environment variable CHIPYARD_DIR is not set. Please set it in env.sh and source env.sh."
  exit
fi

source $CHIPYARD_SOURCE_ENV
conda activate $CHIPYARD_DIR/.conda-env

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
 -DROCKET_TARGET -DSMALL_DATASET \
 -Wl,--wrap,malloc \
 -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany \
 -O2 -ffast-math -fno-common -fno-builtin-printf \
 -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc12 \
 -lm \
 -I$CHIPYARD_DIR/generators/fdra/software/tests//riscv-tests/benchmarks/common \
 -I$CHIPYARD_DIR/generators/fdra/software/tests/riscv-tests \
 -I$CHIPYARD_DIR/generators/fdra/software/tests//riscv-tests/env \
 -I$CHIPYARD_DIR/generators/fdra/software/tests/ \
  -I/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/utilities \
 -T$CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/my_test.ld \
 -DID_STRING= -nostartfiles -static  -DDEFINE_MALLOC \
 -DBAREMETAL=1 -e _start -g  \
 -o $CHIPYARD_DIR/generators/fdra/software/tests/build/bareMetalC/${baremetal_file_name}-baremetal \
 $rootfolder/$main_file \
 $object_files \
 $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/syscalls.c \
 $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/CRunnerUtils.cpp \
 $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/tiny-malloc.c


riscv64-unknown-elf-gcc \
 -DROCKET_TARGET -DSMALL_DATASET \
 -Wl,--wrap,malloc \
 -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany \
 -O2 -ffast-math -fno-common -fno-builtin-printf \
 -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc12 \
 -lm \
 -I$CHIPYARD_DIR/generators/fdra/software/tests//riscv-tests/benchmarks/common \
 -I$CHIPYARD_DIR/generators/fdra/software/tests/riscv-tests \
 -I$CHIPYARD_DIR/generators/fdra/software/tests//riscv-tests/env \
 -I$CHIPYARD_DIR/generators/fdra/software/tests/ \
  -I/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/utilities \
 -T$CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/my_test.ld \
 -DID_STRING= -nostartfiles -static  -DDEFINE_MALLOC \
 -DBAREMETAL=1 -e _start -g  \
 -o $CHIPYARD_DIR/generators/fdra/software/tests/build/bareMetalC/${baremetal_file_name}-baremetal \
 $rootfolder/$main_file \
 $object_files \
 $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/syscalls.c \
 $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/CRunnerUtils.cpp \
 $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/tiny-malloc.c

#  -lgcc  -fno-builtin-malloc -fno-builtin-free  -fno-builtin \
#  $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/tiny-malloc.c \
#  $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/CRunnerUtils.cpp 
#  /home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/utilities/rocket_polybench.c
#  $CHIPYARD_DIR/generators/fdra/software/tests//riscv-tests/benchmarks/common/syscalls.c \
#  $CHIPYARD_DIR/generators/fdra/software/tests/gemm/crt.S \
#  $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/CRunnerUtils.cpp
#  $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/tiny-malloc.c 



#  "$asmfolder"/forward_kernel_1.s \
#  "$asmfolder"/forward_kernel_0.s \
#  "$asmfolder"/syscalls.s \
#  "$asmfolder"/crt.s \
#  "$asmfolder"/forward_kernel_2.s\
#  "$asmfolder"/forward.s \