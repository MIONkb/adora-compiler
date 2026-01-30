source scripts/env.sh
bash scripts/0_compileCtoMLIR.sh 
bash scripts/0_compileCtoMLIR.sh 

cgra-mapper \
    --adg="${CGRA_ADG_PATH}/cgra_adg.json" \
    --op-file="${CGRA_OP_FILE_PATH}/operations.json" \
    --output="/home/jhlou/projects/cocotb/IntVecAddNew/IntVecAdd.py" \
    --output-type="pytest" \
    IR/1_kernels_opt/IntVecAdd_kernel_opt.mlir 