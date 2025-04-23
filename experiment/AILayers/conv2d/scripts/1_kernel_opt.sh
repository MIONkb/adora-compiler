#!/bin/bash
rootfolder=$(pwd)
IRfolder="IR"
srcfolder="$rootfolder/$IRfolder/0_kernels"
tarfolder="$rootfolder/$IRfolder/1_kernels_opt"
tempfolder="$rootfolder/$IRfolder/tempfiles"
echo "current path:$rootfolder"

if [ ! -d "$tarfolder" ]; then
  echo mkdir -p "$tarfolder"
  mkdir -p "$tarfolder"
fi
if [ ! -d "$tempfolder" ]; then
  mkdir -p "$tempfolder"
  echo mkdir -p "$tempfolder"
fi

cd $tarfolder 
rm *.mlir

cd $tempfolder
echo "current path:$tempfolder"
# traverse every file
cnt=0

if [ -z "$CGRA_ADG_PATH" ]; then
  echo "Environment variable CGRA_ADG_PATH is not set. Please set it in env.sh and source env.sh."
  exit
fi

for file in "$srcfolder"/*; do
    filename=$(basename "$file" .mlir)
    echo "$filename"
    if [[ -f "$file" ]]; then
      cgra-opt \
        --ADORA-extract-affine-for-to-kernel \
        --ADORA-simplify-loadstore \
        --adora-math-rewrite \
        --ADORA-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=4 disable-remainder-block explicit-datablock" \
        --ADORA-affine-loop-unroll="cgra-adg=${CGRA_ADG_PATH}/cgra_adg.json" \
        "$file" -o $tarfolder/"$filename"_opt.mlir

        # --ADORA-extract-affine-for-to-kernel \
        # --ADORA-hoist-loadstore \
 # --ADORA-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock" \
        # --ADORA-loop-unroll-jam \
        # --ADORA-affine-loop-unroll="cgra-adg=${CGRA_ADG_PATH}/cgra_adg.json" \
        # 
        # --ADORA-hoist-loadstore \
        # "$file" -o $tarfolder/"$filename"_opt.mlir
      ((cnt++))
      echo $cnt
    fi
done

cd $rootfolder