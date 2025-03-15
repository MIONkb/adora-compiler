#!/bin/bash
rootfolder=$(pwd)
srcfolder="$rootfolder/kernels"
tarfolder="$rootfolder/1_kernels_opt"
tempfolder="$rootfolder/tempfiles"
echo "current path:$rootfolder"

if [ ! -d "$tarfolder" ]; then
  echo mkdir -p "$tarfolder"
  mkdir -p "$tarfolder"
fi
if [ ! -d "$tempfolder" ]; then
  mkdir -p "$tempfolder"
  echo mkdir -p "$tempfolder"
fi


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
        --ADORA-approximate-math \
        --ADORA-extract-affine-for-to-kernel \
        --ADORA-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock" \
        "$file" -o $tarfolder/"$filename"_opt.mlir

               
        # --ADORA-affine-loop-unroll="cgra-adg=${CGRA_ADG_PATH}/cgra_adg.json" \
        # 
        # --ADORA-hoist-loadstore \
        # "$file" -o $tarfolder/"$filename"_opt.mlir
      ((cnt++))
      echo $cnt
    fi
done

cd $rootfolder