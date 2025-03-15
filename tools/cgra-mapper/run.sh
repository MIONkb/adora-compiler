# SPDLOG_LEVEL=trace, debug, info, warn, err, critical, off
cgra-mapper SPDLOG_LEVEL=off\
	--dump-mapped-viz \
	--obj-opt --timeout 3600000 --max-iters 2000 \
	--op-file "/home/tianyi/chipyard/generators/fdra/cgra-mg/src/main/resources/operations.json" \
	--adg "/home/tianyi/chipyard/generators/fdra/cgra-mg/src/main/resources/cgra_adg.json" \
	/home/tianyi/ADORA/app-compiler/cgra-opt/experiments/ResnetSingleBLock/tempfiles/temp.mlir
#	-d "../benchmarks/test/arf/arf.json ../benchmarks/test/centro-fir/centro-fir.json ../benchmarks/test/ewf/ewf.json ../benchmarks/test/fft/fft.json ../benchmarks/test/fir1/fir1.json ../benchmarks/test/resnet1/resnet1.json"
#	-d "../benchmarks/test/arf/arf.json ../benchmarks/test/centro-fir/centro-fir.json ../benchmarks/test/cosine1/cosine1.json ../benchmarks/test/ewf/ewf.json ../benchmarks/test/fir1/fir1.json ../benchmarks/test/resnet1/resnet1.json"
	# -d "../benchmarks/test/fft/fft.json"
  #	-d "../benchmarks/test2/matrixmul/matrixmul.json"
	# -d "../benchmarks/test/cosine2/cosine2.json ../benchmarks/test/fir/fir.json ../benchmarks/test/md/md.json ../benchmarks/test/resnet2/resnet2.json ../benchmarks/test/stencil3d/stencil3d.json"	
	# -d "../benchmarks/test2/conv3/conv3.json ../benchmarks/test2/matrixmul/matrixmul.json ../benchmarks/test2/simple/simple.json"