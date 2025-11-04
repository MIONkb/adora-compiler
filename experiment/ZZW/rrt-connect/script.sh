source scripts/env.sh
bash scripts/0_compileCtoMLIR.sh 
bash scripts/0_compileCtoMLIR.sh 

cgeist \
    -O2 \
    -lm -lgcc \
    -Dsize_t=int -Dwint_t=int -DROCKET_TARGET -D_riscv -DDATA_TYPE_IS_FLOAT \
    --import-all-index \
    -I/home/jhlou/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/include/machine/ \
    -I/home/jhlou/chipyard/.conda-env/riscv-tools/sysroot/usr/include/linux/ \
   main.cpp \
    -S -o IR/rtt_connect.mlir

cgra-mapper \
    --adg="${CGRA_ADG_PATH}/cgra_adg.json" \
    --op-file="${CGRA_OP_FILE_PATH}/operations.json" \
    --output="/home/jhlou/projects/cocotb/IntVecAddNew/IntVecAdd.py" \
    --output-type="pytest" \
    IR/1_kernels_opt/IntVecAdd_kernel_opt.mlir 