#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
#map3 = affine_map<(d0) -> (d0 + 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm(%arg0: memref<?x25xf32>, %arg1: memref<?x30xf32>, %arg2: memref<?x25xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.200000e+00 : f32
    %cst_0 = arith.constant 1.500000e+00 : f32
    affine.for %arg3 = 0 to 20 step 5 {
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
      ADORA.kernel {
        affine.for %arg4 = 0 to 25 {
          %42 = affine.load %0[0, %arg4] : memref<1x25xf32>
          %43 = arith.mulf %42, %cst : f32
          affine.store %43, %1[0, %arg4] : memref<1x25xf32>
          %44 = affine.load %3[0, %arg4] : memref<1x25xf32>
          %45 = arith.mulf %44, %cst : f32
          affine.store %45, %4[0, %arg4] : memref<1x25xf32>
          %46 = affine.load %6[0, %arg4] : memref<1x25xf32>
          %47 = arith.mulf %46, %cst : f32
          affine.store %47, %7[0, %arg4] : memref<1x25xf32>
          %48 = affine.load %9[0, %arg4] : memref<1x25xf32>
          %49 = arith.mulf %48, %cst : f32
          affine.store %49, %10[0, %arg4] : memref<1x25xf32>
          %50 = affine.load %12[0, %arg4] : memref<1x25xf32>
          %51 = arith.mulf %50, %cst : f32
          affine.store %51, %13[0, %arg4] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_gemm_0"}
      ADORA.BlockStore %1, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %14 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %15 = ADORA.BlockLoad %arg1 [%arg3, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %16 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %17 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %18 = affine.apply #map(%arg3)
      ADORA.BlockStore %4, %arg0 [%18, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %19 = ADORA.BlockLoad %arg0 [%18, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %20 = ADORA.BlockLoad %arg1 [%18, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %21 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %22 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %23 = affine.apply #map1(%arg3)
      ADORA.BlockStore %7, %arg0 [%23, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %24 = ADORA.BlockLoad %arg0 [%23, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %25 = ADORA.BlockLoad %arg1 [%23, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %26 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %27 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %28 = affine.apply #map2(%arg3)
      ADORA.BlockStore %10, %arg0 [%28, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %29 = ADORA.BlockLoad %arg0 [%28, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %30 = ADORA.BlockLoad %arg1 [%28, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %31 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %32 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %33 = affine.apply #map3(%arg3)
      ADORA.BlockStore %13, %arg0 [%33, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %34 = ADORA.BlockLoad %arg0 [%33, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %35 = ADORA.BlockLoad %arg1 [%33, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %36 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %37 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      ADORA.kernel {
        affine.for %arg4 = 0 to 25 {
          %42 = affine.load %14[0, %arg4] : memref<1x25xf32>
          %43 = affine.apply #map(%arg3)
          %44 = affine.load %19[0, %arg4] : memref<1x25xf32>
          %45 = affine.apply #map1(%arg3)
          %46 = affine.load %24[0, %arg4] : memref<1x25xf32>
          %47 = affine.apply #map2(%arg3)
          %48 = affine.load %29[0, %arg4] : memref<1x25xf32>
          %49 = affine.apply #map3(%arg3)
          %50 = affine.load %34[0, %arg4] : memref<1x25xf32>
          %51:5 = affine.for %arg5 = 0 to 30 iter_args(%arg6 = %42, %arg7 = %44, %arg8 = %46, %arg9 = %48, %arg10 = %50) -> (f32, f32, f32, f32, f32) {
            %56 = affine.load %15[0, %arg5] : memref<1x30xf32>
            %57 = arith.mulf %56, %cst_0 : f32
            %58 = affine.load %16[%arg5, %arg4] : memref<30x25xf32>
            %59 = arith.mulf %57, %58 : f32
            %60 = arith.addf %arg6, %59 : f32
            %61 = affine.apply #map(%arg3)
            %62 = affine.load %20[0, %arg5] : memref<1x30xf32>
            %63 = arith.mulf %62, %cst_0 : f32
            %64 = affine.load %21[%arg5, %arg4] : memref<30x25xf32>
            %65 = arith.mulf %63, %64 : f32
            %66 = arith.addf %arg7, %65 : f32
            %67 = affine.apply #map1(%arg3)
            %68 = affine.load %25[0, %arg5] : memref<1x30xf32>
            %69 = arith.mulf %68, %cst_0 : f32
            %70 = affine.load %26[%arg5, %arg4] : memref<30x25xf32>
            %71 = arith.mulf %69, %70 : f32
            %72 = arith.addf %arg8, %71 : f32
            %73 = affine.apply #map2(%arg3)
            %74 = affine.load %30[0, %arg5] : memref<1x30xf32>
            %75 = arith.mulf %74, %cst_0 : f32
            %76 = affine.load %31[%arg5, %arg4] : memref<30x25xf32>
            %77 = arith.mulf %75, %76 : f32
            %78 = arith.addf %arg9, %77 : f32
            %79 = affine.apply #map3(%arg3)
            %80 = affine.load %35[0, %arg5] : memref<1x30xf32>
            %81 = arith.mulf %80, %cst_0 : f32
            %82 = affine.load %36[%arg5, %arg4] : memref<30x25xf32>
            %83 = arith.mulf %81, %82 : f32
            %84 = arith.addf %arg10, %83 : f32
            affine.yield %60, %66, %72, %78, %84 : f32, f32, f32, f32, f32
          }
          affine.store %51#0, %17[0, %arg4] : memref<1x25xf32>
          %52 = affine.apply #map(%arg3)
          affine.store %51#1, %22[0, %arg4] : memref<1x25xf32>
          %53 = affine.apply #map1(%arg3)
          affine.store %51#2, %27[0, %arg4] : memref<1x25xf32>
          %54 = affine.apply #map2(%arg3)
          affine.store %51#3, %32[0, %arg4] : memref<1x25xf32>
          %55 = affine.apply #map3(%arg3)
          affine.store %51#4, %37[0, %arg4] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_gemm_1"}
      ADORA.BlockStore %17, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %38 = affine.apply #map(%arg3)
      ADORA.BlockStore %22, %arg0 [%38, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %39 = affine.apply #map1(%arg3)
      ADORA.BlockStore %27, %arg0 [%39, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %40 = affine.apply #map2(%arg3)
      ADORA.BlockStore %32, %arg0 [%40, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %41 = affine.apply #map3(%arg3)
      ADORA.BlockStore %37, %arg0 [%41, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
    }
    return
  }
}
