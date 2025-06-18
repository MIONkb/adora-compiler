#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
#map3 = affine_map<(d0) -> (d0 + 4)>
#map4 = affine_map<(d0) -> (d0 + 5)>
#map5 = affine_map<(d0) -> (d0 + 6)>
#map6 = affine_map<(d0) -> (d0 + 7)>
#map7 = affine_map<(d0) -> (d0 + 8)>
#map8 = affine_map<(d0) -> (d0 + 9)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm(%arg0: memref<?x25xf32>, %arg1: memref<?x30xf32>, %arg2: memref<?x25xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.200000e+00 : f32
    %cst_0 = arith.constant 1.500000e+00 : f32
    affine.for %arg3 = 0 to 20 step 10 {
      %0 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
      %1 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %2 = affine.apply #map(%arg3)
      %3 = ADORA.BlockLoad %arg0 [%2, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
      %4 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %5 = affine.apply #map1(%arg3)
      %6 = ADORA.BlockLoad %arg0 [%5, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
      %7 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %8 = affine.apply #map2(%arg3)
      %9 = ADORA.BlockLoad %arg0 [%8, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
      %10 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %11 = affine.apply #map3(%arg3)
      %12 = ADORA.BlockLoad %arg0 [%11, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
      %13 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %14 = affine.apply #map4(%arg3)
      %15 = ADORA.BlockLoad %arg0 [%14, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
      %16 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %17 = affine.apply #map5(%arg3)
      %18 = ADORA.BlockLoad %arg0 [%17, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
      %19 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %20 = affine.apply #map6(%arg3)
      %21 = ADORA.BlockLoad %arg0 [%20, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
      %22 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %23 = affine.apply #map7(%arg3)
      %24 = ADORA.BlockLoad %arg0 [%23, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
      %25 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %26 = affine.apply #map8(%arg3)
      %27 = ADORA.BlockLoad %arg0 [%26, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
      %28 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      ADORA.kernel {
        affine.for %arg4 = 0 to 25 {
          %87 = affine.load %0[0, %arg4] : memref<1x25xf32>
          %88 = arith.mulf %87, %cst : f32
          affine.store %88, %1[0, %arg4] : memref<1x25xf32>
          %89 = affine.load %3[0, %arg4] : memref<1x25xf32>
          %90 = arith.mulf %89, %cst : f32
          affine.store %90, %4[0, %arg4] : memref<1x25xf32>
          %91 = affine.load %6[0, %arg4] : memref<1x25xf32>
          %92 = arith.mulf %91, %cst : f32
          affine.store %92, %7[0, %arg4] : memref<1x25xf32>
          %93 = affine.load %9[0, %arg4] : memref<1x25xf32>
          %94 = arith.mulf %93, %cst : f32
          affine.store %94, %10[0, %arg4] : memref<1x25xf32>
          %95 = affine.load %12[0, %arg4] : memref<1x25xf32>
          %96 = arith.mulf %95, %cst : f32
          affine.store %96, %13[0, %arg4] : memref<1x25xf32>
          %97 = affine.load %15[0, %arg4] : memref<1x25xf32>
          %98 = arith.mulf %97, %cst : f32
          affine.store %98, %16[0, %arg4] : memref<1x25xf32>
          %99 = affine.load %18[0, %arg4] : memref<1x25xf32>
          %100 = arith.mulf %99, %cst : f32
          affine.store %100, %19[0, %arg4] : memref<1x25xf32>
          %101 = affine.load %21[0, %arg4] : memref<1x25xf32>
          %102 = arith.mulf %101, %cst : f32
          affine.store %102, %22[0, %arg4] : memref<1x25xf32>
          %103 = affine.load %24[0, %arg4] : memref<1x25xf32>
          %104 = arith.mulf %103, %cst : f32
          affine.store %104, %25[0, %arg4] : memref<1x25xf32>
          %105 = affine.load %27[0, %arg4] : memref<1x25xf32>
          %106 = arith.mulf %105, %cst : f32
          affine.store %106, %28[0, %arg4] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_gemm_0"}
      ADORA.BlockStore %1, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %29 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %30 = ADORA.BlockLoad %arg1 [%arg3, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %31 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %32 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %33 = affine.apply #map(%arg3)
      ADORA.BlockStore %4, %arg0 [%33, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %34 = ADORA.BlockLoad %arg0 [%33, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %35 = ADORA.BlockLoad %arg1 [%33, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %36 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %37 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %38 = affine.apply #map1(%arg3)
      ADORA.BlockStore %7, %arg0 [%38, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %39 = ADORA.BlockLoad %arg0 [%38, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %40 = ADORA.BlockLoad %arg1 [%38, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %41 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %42 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %43 = affine.apply #map2(%arg3)
      ADORA.BlockStore %10, %arg0 [%43, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %44 = ADORA.BlockLoad %arg0 [%43, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %45 = ADORA.BlockLoad %arg1 [%43, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %46 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %47 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %48 = affine.apply #map3(%arg3)
      ADORA.BlockStore %13, %arg0 [%48, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %49 = ADORA.BlockLoad %arg0 [%48, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %50 = ADORA.BlockLoad %arg1 [%48, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %51 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %52 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %53 = affine.apply #map4(%arg3)
      ADORA.BlockStore %16, %arg0 [%53, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %54 = ADORA.BlockLoad %arg0 [%53, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %55 = ADORA.BlockLoad %arg1 [%53, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %56 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %57 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %58 = affine.apply #map5(%arg3)
      ADORA.BlockStore %19, %arg0 [%58, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %59 = ADORA.BlockLoad %arg0 [%58, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %60 = ADORA.BlockLoad %arg1 [%58, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %61 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %62 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %63 = affine.apply #map6(%arg3)
      ADORA.BlockStore %22, %arg0 [%63, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %64 = ADORA.BlockLoad %arg0 [%63, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %65 = ADORA.BlockLoad %arg1 [%63, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %66 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %67 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %68 = affine.apply #map7(%arg3)
      ADORA.BlockStore %25, %arg0 [%68, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %69 = ADORA.BlockLoad %arg0 [%68, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %70 = ADORA.BlockLoad %arg1 [%68, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %71 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %72 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %73 = affine.apply #map8(%arg3)
      ADORA.BlockStore %28, %arg0 [%73, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %74 = ADORA.BlockLoad %arg0 [%73, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %75 = ADORA.BlockLoad %arg1 [%73, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %76 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %77 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      ADORA.kernel {
        affine.for %arg4 = 0 to 25 {
          %87 = affine.load %29[0, %arg4] : memref<1x25xf32>
          %88 = affine.apply #map(%arg3)
          %89 = affine.load %34[0, %arg4] : memref<1x25xf32>
          %90 = affine.apply #map1(%arg3)
          %91 = affine.load %39[0, %arg4] : memref<1x25xf32>
          %92 = affine.apply #map2(%arg3)
          %93 = affine.load %44[0, %arg4] : memref<1x25xf32>
          %94 = affine.apply #map3(%arg3)
          %95 = affine.load %49[0, %arg4] : memref<1x25xf32>
          %96 = affine.apply #map4(%arg3)
          %97 = affine.load %54[0, %arg4] : memref<1x25xf32>
          %98 = affine.apply #map5(%arg3)
          %99 = affine.load %59[0, %arg4] : memref<1x25xf32>
          %100 = affine.apply #map6(%arg3)
          %101 = affine.load %64[0, %arg4] : memref<1x25xf32>
          %102 = affine.apply #map7(%arg3)
          %103 = affine.load %69[0, %arg4] : memref<1x25xf32>
          %104 = affine.apply #map8(%arg3)
          %105 = affine.load %74[0, %arg4] : memref<1x25xf32>
          %106:10 = affine.for %arg5 = 0 to 30 iter_args(%arg6 = %87, %arg7 = %89, %arg8 = %91, %arg9 = %93, %arg10 = %95, %arg11 = %97, %arg12 = %99, %arg13 = %101, %arg14 = %103, %arg15 = %105) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
            %116 = affine.load %30[0, %arg5] : memref<1x30xf32>
            %117 = arith.mulf %116, %cst_0 : f32
            %118 = affine.load %31[%arg5, %arg4] : memref<30x25xf32>
            %119 = arith.mulf %117, %118 : f32
            %120 = arith.addf %arg6, %119 : f32
            %121 = affine.apply #map(%arg3)
            %122 = affine.load %35[0, %arg5] : memref<1x30xf32>
            %123 = arith.mulf %122, %cst_0 : f32
            %124 = affine.load %36[%arg5, %arg4] : memref<30x25xf32>
            %125 = arith.mulf %123, %124 : f32
            %126 = arith.addf %arg7, %125 : f32
            %127 = affine.apply #map1(%arg3)
            %128 = affine.load %40[0, %arg5] : memref<1x30xf32>
            %129 = arith.mulf %128, %cst_0 : f32
            %130 = affine.load %41[%arg5, %arg4] : memref<30x25xf32>
            %131 = arith.mulf %129, %130 : f32
            %132 = arith.addf %arg8, %131 : f32
            %133 = affine.apply #map2(%arg3)
            %134 = affine.load %45[0, %arg5] : memref<1x30xf32>
            %135 = arith.mulf %134, %cst_0 : f32
            %136 = affine.load %46[%arg5, %arg4] : memref<30x25xf32>
            %137 = arith.mulf %135, %136 : f32
            %138 = arith.addf %arg9, %137 : f32
            %139 = affine.apply #map3(%arg3)
            %140 = affine.load %50[0, %arg5] : memref<1x30xf32>
            %141 = arith.mulf %140, %cst_0 : f32
            %142 = affine.load %51[%arg5, %arg4] : memref<30x25xf32>
            %143 = arith.mulf %141, %142 : f32
            %144 = arith.addf %arg10, %143 : f32
            %145 = affine.apply #map4(%arg3)
            %146 = affine.load %55[0, %arg5] : memref<1x30xf32>
            %147 = arith.mulf %146, %cst_0 : f32
            %148 = affine.load %56[%arg5, %arg4] : memref<30x25xf32>
            %149 = arith.mulf %147, %148 : f32
            %150 = arith.addf %arg11, %149 : f32
            %151 = affine.apply #map5(%arg3)
            %152 = affine.load %60[0, %arg5] : memref<1x30xf32>
            %153 = arith.mulf %152, %cst_0 : f32
            %154 = affine.load %61[%arg5, %arg4] : memref<30x25xf32>
            %155 = arith.mulf %153, %154 : f32
            %156 = arith.addf %arg12, %155 : f32
            %157 = affine.apply #map6(%arg3)
            %158 = affine.load %65[0, %arg5] : memref<1x30xf32>
            %159 = arith.mulf %158, %cst_0 : f32
            %160 = affine.load %66[%arg5, %arg4] : memref<30x25xf32>
            %161 = arith.mulf %159, %160 : f32
            %162 = arith.addf %arg13, %161 : f32
            %163 = affine.apply #map7(%arg3)
            %164 = affine.load %70[0, %arg5] : memref<1x30xf32>
            %165 = arith.mulf %164, %cst_0 : f32
            %166 = affine.load %71[%arg5, %arg4] : memref<30x25xf32>
            %167 = arith.mulf %165, %166 : f32
            %168 = arith.addf %arg14, %167 : f32
            %169 = affine.apply #map8(%arg3)
            %170 = affine.load %75[0, %arg5] : memref<1x30xf32>
            %171 = arith.mulf %170, %cst_0 : f32
            %172 = affine.load %76[%arg5, %arg4] : memref<30x25xf32>
            %173 = arith.mulf %171, %172 : f32
            %174 = arith.addf %arg15, %173 : f32
            affine.yield %120, %126, %132, %138, %144, %150, %156, %162, %168, %174 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
          }
          affine.store %106#0, %32[0, %arg4] : memref<1x25xf32>
          %107 = affine.apply #map(%arg3)
          affine.store %106#1, %37[0, %arg4] : memref<1x25xf32>
          %108 = affine.apply #map1(%arg3)
          affine.store %106#2, %42[0, %arg4] : memref<1x25xf32>
          %109 = affine.apply #map2(%arg3)
          affine.store %106#3, %47[0, %arg4] : memref<1x25xf32>
          %110 = affine.apply #map3(%arg3)
          affine.store %106#4, %52[0, %arg4] : memref<1x25xf32>
          %111 = affine.apply #map4(%arg3)
          affine.store %106#5, %57[0, %arg4] : memref<1x25xf32>
          %112 = affine.apply #map5(%arg3)
          affine.store %106#6, %62[0, %arg4] : memref<1x25xf32>
          %113 = affine.apply #map6(%arg3)
          affine.store %106#7, %67[0, %arg4] : memref<1x25xf32>
          %114 = affine.apply #map7(%arg3)
          affine.store %106#8, %72[0, %arg4] : memref<1x25xf32>
          %115 = affine.apply #map8(%arg3)
          affine.store %106#9, %77[0, %arg4] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_gemm_1"}
      ADORA.BlockStore %32, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %78 = affine.apply #map(%arg3)
      ADORA.BlockStore %37, %arg0 [%78, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %79 = affine.apply #map1(%arg3)
      ADORA.BlockStore %42, %arg0 [%79, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %80 = affine.apply #map2(%arg3)
      ADORA.BlockStore %47, %arg0 [%80, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %81 = affine.apply #map3(%arg3)
      ADORA.BlockStore %52, %arg0 [%81, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %82 = affine.apply #map4(%arg3)
      ADORA.BlockStore %57, %arg0 [%82, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %83 = affine.apply #map5(%arg3)
      ADORA.BlockStore %62, %arg0 [%83, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %84 = affine.apply #map6(%arg3)
      ADORA.BlockStore %67, %arg0 [%84, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %85 = affine.apply #map7(%arg3)
      ADORA.BlockStore %72, %arg0 [%85, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %86 = affine.apply #map8(%arg3)
      ADORA.BlockStore %77, %arg0 [%86, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
    }
    return
  }
}
