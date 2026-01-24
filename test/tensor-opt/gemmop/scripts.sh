
tensor-opt \
  --adora-gemm-op-strategy-decision="adg-fn=../../../../../rtl/spec/vitra_cgra_adg.json bus-bandwidth=8" \
  gemm_init.mlir > gemm_optimal.mlir
tensor-opt --adora-gen-tensor-op-cdfg gemm_optimal.mlir > dfg.mlir

cgra-mapper     \
  --adg="../../../../../rtl/spec/vitra_cgra_adg.json"     \
  --op-file="../../../../../rtl/spec/operations.json"     \
  --output="../matmul_0.py"     \
  --output-type="pytest"     --obj-opt=true     --max-iters=2000     \
  "gemm_optimal.mlir"

### if you want a faster but bad latency mapping, use following:
# cgra-mapper     \
#   --adg="../../../../rtl/spec/vitra_cgra_adg.json"     \
#   --op-file="../../../../rtl/spec/operations.json"     \
#   --output="../matmul_0.py"     \
#   --output-type="pytest"     --obj-opt=flase     --max-iters=50     \
#   "gemm_withnotensor.mlir"