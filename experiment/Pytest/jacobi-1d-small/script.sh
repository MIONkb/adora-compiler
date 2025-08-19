source scripts/env.sh
bash scripts/0_compileCtoMLIR.sh 
bash scripts/0_compileCtoMLIR.sh 

cgra-mapper \
    --adg="${CGRA_ADG_PATH}/cgra_adg.json" \
    --op-file="${CGRA_OP_FILE_PATH}/operations.json" \
    --output="/home/jhlou/CGRVOPT/cgra-opt/experiment/Pytest/jacobi-1d-small/jacobi-1d-small.py" \
    --output-type="pytest" \
    /home/jhlou/CGRVOPT/cgra-opt/experiment/Pytest/jacobi-1d-small/1_kernels_opt/jacobi_1d_kernel_0_opt.mlir