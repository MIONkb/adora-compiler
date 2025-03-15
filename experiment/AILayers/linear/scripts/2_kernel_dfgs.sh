#!/bin/bash

rootfolder=$(pwd)
srcfolder="$rootfolder/1_kernels_opt"
tarfolder="$rootfolder/2_dfgs"
tempfolder="$rootfolder/tempfiles"
kernel_basename="forward_kernel"

if [ ! -d "$tarfolder" ]; then
  mkdir -p "$tarfolder"
  echo mkdir -p "$tarfolder"
fi
if [ ! -d "$tempfolder" ]; then
  mkdir -p "$tempfolder"
  echo mkdir -p "$tempfolder"
fi
cd $tempfolder
if [[ "$(pwd)" == "$tempfolder" ]]; then
  rm *.dot *.ll *.json *.txt
  cd -
fi

cd $tarfolder
if [[ "$(pwd)" == "$tarfolder" ]]; then
  rm *.dot
  cd -
fi

cd $rootfolder
# traverse every file
cnt=0

cd $tempfolder
for file in "$srcfolder"/*.mlir; do
    filename=$(basename "$file" _opt.mlir)
    echo "$filename"
    if [[ -f "$file" ]]; then
      cgra-opt\
        --ADORA-kernel-dfg-gen \
        $file 
      
      cp ./"$filename"_CDFG.dot $tarfolder/"$filename"_CDFG.dot
      dot $tarfolder/"$filename"_CDFG.dot -Tpng -o $tarfolder/"$filename"_CDFG.png

      ((cnt++))
      echo $cnt
    fi
done
cnt=0
cd -
