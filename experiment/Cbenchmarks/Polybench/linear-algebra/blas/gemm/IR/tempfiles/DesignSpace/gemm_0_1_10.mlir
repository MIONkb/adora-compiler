#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
#map3 = affine_map<(d0) -> (d0 + 4)>
#map4 = affine_map<(d0) -> (d0 + 5)>
#map5 = affine_map<(d0) -> (d0 + 6)>
#map6 = affine_map<(d0) -> (d0 + 7)>
#map7 = affine_map<(d0) -> (d0 + 8)>
#map8 = affine_map<(d0) -> (d0 + 9)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm(%arg0: f32, %arg1: f32, %arg2: memref<?x25xf32>, %arg3: memref<?x30xf32>, %arg4: memref<?x25xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg5 = 0 to 20 step 10 {
      %0 = ADORA.BlockLoad %arg2 [%arg5, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %1 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %2 = affine.apply #map(%arg5)
      %3 = ADORA.BlockLoad %arg2 [%2, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %4 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %5 = affine.apply #map1(%arg5)
      %6 = ADORA.BlockLoad %arg2 [%5, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %7 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %8 = affine.apply #map2(%arg5)
      %9 = ADORA.BlockLoad %arg2 [%8, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %10 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %11 = affine.apply #map3(%arg5)
      %12 = ADORA.BlockLoad %arg2 [%11, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %13 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %14 = affine.apply #map4(%arg5)
      %15 = ADORA.BlockLoad %arg2 [%14, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %16 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %17 = affine.apply #map5(%arg5)
      %18 = ADORA.BlockLoad %arg2 [%17, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %19 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %20 = affine.apply #map6(%arg5)
      %21 = ADORA.BlockLoad %arg2 [%20, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %22 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %23 = affine.apply #map7(%arg5)
      %24 = ADORA.BlockLoad %arg2 [%23, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %25 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %26 = affine.apply #map8(%arg5)
      %27 = ADORA.BlockLoad %arg2 [%26, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %28 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      ADORA.kernel {
        affine.for %arg6 = 0 to 25 {
          %87 = affine.load %0[0, %arg6] : memref<1x25xf32>
          %88 = arith.mulf %87, %arg1 : f32
          affine.store %88, %1[0, %arg6] : memref<1x25xf32>
          %89 = affine.load %3[0, %arg6] : memref<1x25xf32>
          %90 = arith.mulf %89, %arg1 : f32
          affine.store %90, %4[0, %arg6] : memref<1x25xf32>
          %91 = affine.load %6[0, %arg6] : memref<1x25xf32>
          %92 = arith.mulf %91, %arg1 : f32
          affine.store %92, %7[0, %arg6] : memref<1x25xf32>
          %93 = affine.load %9[0, %arg6] : memref<1x25xf32>
          %94 = arith.mulf %93, %arg1 : f32
          affine.store %94, %10[0, %arg6] : memref<1x25xf32>
          %95 = affine.load %12[0, %arg6] : memref<1x25xf32>
          %96 = arith.mulf %95, %arg1 : f32
          affine.store %96, %13[0, %arg6] : memref<1x25xf32>
          %97 = affine.load %15[0, %arg6] : memref<1x25xf32>
          %98 = arith.mulf %97, %arg1 : f32
          affine.store %98, %16[0, %arg6] : memref<1x25xf32>
          %99 = affine.load %18[0, %arg6] : memref<1x25xf32>
          %100 = arith.mulf %99, %arg1 : f32
          affine.store %100, %19[0, %arg6] : memref<1x25xf32>
          %101 = affine.load %21[0, %arg6] : memref<1x25xf32>
          %102 = arith.mulf %101, %arg1 : f32
          affine.store %102, %22[0, %arg6] : memref<1x25xf32>
          %103 = affine.load %24[0, %arg6] : memref<1x25xf32>
          %104 = arith.mulf %103, %arg1 : f32
          affine.store %104, %25[0, %arg6] : memref<1x25xf32>
          %105 = affine.load %27[0, %arg6] : memref<1x25xf32>
          %106 = arith.mulf %105, %arg1 : f32
          affine.store %106, %28[0, %arg6] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "gemm_0"}
      ADORA.BlockStore %1, %arg2 [%arg5, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %29 = ADORA.BlockLoad %arg3 [%arg5, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %30 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %31 = ADORA.BlockLoad %arg2 [%arg5, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %32 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %33 = affine.apply #map(%arg5)
      ADORA.BlockStore %4, %arg2 [%33, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %34 = ADORA.BlockLoad %arg3 [%33, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %35 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %36 = ADORA.BlockLoad %arg2 [%33, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %37 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %38 = affine.apply #map1(%arg5)
      ADORA.BlockStore %7, %arg2 [%38, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %39 = ADORA.BlockLoad %arg3 [%38, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %40 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %41 = ADORA.BlockLoad %arg2 [%38, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %42 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %43 = affine.apply #map2(%arg5)
      ADORA.BlockStore %10, %arg2 [%43, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %44 = ADORA.BlockLoad %arg3 [%43, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %45 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %46 = ADORA.BlockLoad %arg2 [%43, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %47 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %48 = affine.apply #map3(%arg5)
      ADORA.BlockStore %13, %arg2 [%48, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %49 = ADORA.BlockLoad %arg3 [%48, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %50 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %51 = ADORA.BlockLoad %arg2 [%48, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %52 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %53 = affine.apply #map4(%arg5)
      ADORA.BlockStore %16, %arg2 [%53, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %54 = ADORA.BlockLoad %arg3 [%53, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %55 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %56 = ADORA.BlockLoad %arg2 [%53, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %57 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %58 = affine.apply #map5(%arg5)
      ADORA.BlockStore %19, %arg2 [%58, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %59 = ADORA.BlockLoad %arg3 [%58, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %60 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %61 = ADORA.BlockLoad %arg2 [%58, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %62 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %63 = affine.apply #map6(%arg5)
      ADORA.BlockStore %22, %arg2 [%63, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %64 = ADORA.BlockLoad %arg3 [%63, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %65 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %66 = ADORA.BlockLoad %arg2 [%63, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %67 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %68 = affine.apply #map7(%arg5)
      ADORA.BlockStore %25, %arg2 [%68, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %69 = ADORA.BlockLoad %arg3 [%68, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %70 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %71 = ADORA.BlockLoad %arg2 [%68, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %72 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %73 = affine.apply #map8(%arg5)
      ADORA.BlockStore %28, %arg2 [%73, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %74 = ADORA.BlockLoad %arg3 [%73, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %75 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %76 = ADORA.BlockLoad %arg2 [%73, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %77 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      ADORA.kernel {
        affine.for %arg6 = 0 to 30 {
          affine.for %arg7 = 0 to 25 {
            %87 = affine.load %29[0, %arg6] : memref<1x30xf32>
            %88 = arith.mulf %arg0, %87 : f32
            %89 = affine.load %30[%arg6, %arg7] : memref<30x25xf32>
            %90 = arith.mulf %88, %89 : f32
            %91 = affine.load %31[0, %arg7] : memref<1x25xf32>
            %92 = arith.addf %91, %90 : f32
            affine.store %92, %32[0, %arg7] : memref<1x25xf32>
            %93 = affine.apply #map(%arg5)
            %94 = affine.load %34[0, %arg6] : memref<1x30xf32>
            %95 = arith.mulf %arg0, %94 : f32
            %96 = affine.load %35[%arg6, %arg7] : memref<30x25xf32>
            %97 = arith.mulf %95, %96 : f32
            %98 = affine.load %36[0, %arg7] : memref<1x25xf32>
            %99 = arith.addf %98, %97 : f32
            affine.store %99, %37[0, %arg7] : memref<1x25xf32>
            %100 = affine.apply #map1(%arg5)
            %101 = affine.load %39[0, %arg6] : memref<1x30xf32>
            %102 = arith.mulf %arg0, %101 : f32
            %103 = affine.load %40[%arg6, %arg7] : memref<30x25xf32>
            %104 = arith.mulf %102, %103 : f32
            %105 = affine.load %41[0, %arg7] : memref<1x25xf32>
            %106 = arith.addf %105, %104 : f32
            affine.store %106, %42[0, %arg7] : memref<1x25xf32>
            %107 = affine.apply #map2(%arg5)
            %108 = affine.load %44[0, %arg6] : memref<1x30xf32>
            %109 = arith.mulf %arg0, %108 : f32
            %110 = affine.load %45[%arg6, %arg7] : memref<30x25xf32>
            %111 = arith.mulf %109, %110 : f32
            %112 = affine.load %46[0, %arg7] : memref<1x25xf32>
            %113 = arith.addf %112, %111 : f32
            affine.store %113, %47[0, %arg7] : memref<1x25xf32>
            %114 = affine.apply #map3(%arg5)
            %115 = affine.load %49[0, %arg6] : memref<1x30xf32>
            %116 = arith.mulf %arg0, %115 : f32
            %117 = affine.load %50[%arg6, %arg7] : memref<30x25xf32>
            %118 = arith.mulf %116, %117 : f32
            %119 = affine.load %51[0, %arg7] : memref<1x25xf32>
            %120 = arith.addf %119, %118 : f32
            affine.store %120, %52[0, %arg7] : memref<1x25xf32>
            %121 = affine.apply #map4(%arg5)
            %122 = affine.load %54[0, %arg6] : memref<1x30xf32>
            %123 = arith.mulf %arg0, %122 : f32
            %124 = affine.load %55[%arg6, %arg7] : memref<30x25xf32>
            %125 = arith.mulf %123, %124 : f32
            %126 = affine.load %56[0, %arg7] : memref<1x25xf32>
            %127 = arith.addf %126, %125 : f32
            affine.store %127, %57[0, %arg7] : memref<1x25xf32>
            %128 = affine.apply #map5(%arg5)
            %129 = affine.load %59[0, %arg6] : memref<1x30xf32>
            %130 = arith.mulf %arg0, %129 : f32
            %131 = affine.load %60[%arg6, %arg7] : memref<30x25xf32>
            %132 = arith.mulf %130, %131 : f32
            %133 = affine.load %61[0, %arg7] : memref<1x25xf32>
            %134 = arith.addf %133, %132 : f32
            affine.store %134, %62[0, %arg7] : memref<1x25xf32>
            %135 = affine.apply #map6(%arg5)
            %136 = affine.load %64[0, %arg6] : memref<1x30xf32>
            %137 = arith.mulf %arg0, %136 : f32
            %138 = affine.load %65[%arg6, %arg7] : memref<30x25xf32>
            %139 = arith.mulf %137, %138 : f32
            %140 = affine.load %66[0, %arg7] : memref<1x25xf32>
            %141 = arith.addf %140, %139 : f32
            affine.store %141, %67[0, %arg7] : memref<1x25xf32>
            %142 = affine.apply #map7(%arg5)
            %143 = affine.load %69[0, %arg6] : memref<1x30xf32>
            %144 = arith.mulf %arg0, %143 : f32
            %145 = affine.load %70[%arg6, %arg7] : memref<30x25xf32>
            %146 = arith.mulf %144, %145 : f32
            %147 = affine.load %71[0, %arg7] : memref<1x25xf32>
            %148 = arith.addf %147, %146 : f32
            affine.store %148, %72[0, %arg7] : memref<1x25xf32>
            %149 = affine.apply #map8(%arg5)
            %150 = affine.load %74[0, %arg6] : memref<1x30xf32>
            %151 = arith.mulf %arg0, %150 : f32
            %152 = affine.load %75[%arg6, %arg7] : memref<30x25xf32>
            %153 = arith.mulf %151, %152 : f32
            %154 = affine.load %76[0, %arg7] : memref<1x25xf32>
            %155 = arith.addf %154, %153 : f32
            affine.store %155, %77[0, %arg7] : memref<1x25xf32>
          }
        }
        ADORA.terminator
      } {KernelName = "gemm_1"}
      ADORA.BlockStore %32, %arg2 [%arg5, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %78 = affine.apply #map(%arg5)
      ADORA.BlockStore %37, %arg2 [%78, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %79 = affine.apply #map1(%arg5)
      ADORA.BlockStore %42, %arg2 [%79, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %80 = affine.apply #map2(%arg5)
      ADORA.BlockStore %47, %arg2 [%80, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %81 = affine.apply #map3(%arg5)
      ADORA.BlockStore %52, %arg2 [%81, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %82 = affine.apply #map4(%arg5)
      ADORA.BlockStore %57, %arg2 [%82, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %83 = affine.apply #map5(%arg5)
      ADORA.BlockStore %62, %arg2 [%83, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %84 = affine.apply #map6(%arg5)
      ADORA.BlockStore %67, %arg2 [%84, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %85 = affine.apply #map7(%arg5)
      ADORA.BlockStore %72, %arg2 [%85, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %86 = affine.apply #map8(%arg5)
      ADORA.BlockStore %77, %arg2 [%86, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
    }
    return
  }
}
