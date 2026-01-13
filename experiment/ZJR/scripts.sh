cgeist  -O3 gray.c  -S -o gray.mlir && adoracc.py gray.mlir

cgra-opt \
    --adora-extract-affine-for-to-kernel \
    --adora-simplify-loadstore \
    --adora-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock" \
    --canonicalize \
    --adora-auto-unroll="cgra-adg=/home/jhlou/CGRVOPT/MatrixMeld/vitrartl/spec/vitra_cgra_adg.json" \
    gray.mlir -o gray_opt.mlir

    # --adora-auto-unroll="cgra-adg=/home/jhlou/CGRVOPT/MatrixMeld/vitrartl/spec/vitra_cgra_adg.json" \

cgra-opt \
    --adora-auto-unroll="cgra-adg=/home/jhlou/CGRVOPT/MatrixMeld/vitrartl/spec/vitra_cgra_adg.json" \
    tiled.mlir -o gray_opt.mlir

cgra-opt \
    --adora-kernel-dfg-gen \
    gray_opt.mlir

