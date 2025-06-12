#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm(%arg0: memref<?x25xf32>, %arg1: memref<?x30xf32>, %arg2: memref<?x25xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.200000e+00 : f32
    %cst_0 = arith.constant 1.500000e+00 : f32
    affine.for %arg3 = 0 to 20 step 4 {
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
      ADORA.kernel {
        affine.for %arg4 = 0 to 25 {
          %33 = affine.load %0[0, %arg4] : memref<1x25xf32>
          %34 = arith.mulf %33, %cst : f32
          affine.store %34, %1[0, %arg4] : memref<1x25xf32>
          %35 = affine.load %3[0, %arg4] : memref<1x25xf32>
          %36 = arith.mulf %35, %cst : f32
          affine.store %36, %4[0, %arg4] : memref<1x25xf32>
          %37 = affine.load %6[0, %arg4] : memref<1x25xf32>
          %38 = arith.mulf %37, %cst : f32
          affine.store %38, %7[0, %arg4] : memref<1x25xf32>
          %39 = affine.load %9[0, %arg4] : memref<1x25xf32>
          %40 = arith.mulf %39, %cst : f32
          affine.store %40, %10[0, %arg4] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_gemm_0"}
      ADORA.BlockStore %1, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %11 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %12 = ADORA.BlockLoad %arg1 [%arg3, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %13 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %14 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %15 = affine.apply #map(%arg3)
      ADORA.BlockStore %4, %arg0 [%15, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %16 = ADORA.BlockLoad %arg0 [%15, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %17 = ADORA.BlockLoad %arg1 [%15, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %18 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %19 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %20 = affine.apply #map1(%arg3)
      ADORA.BlockStore %7, %arg0 [%20, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %21 = ADORA.BlockLoad %arg0 [%20, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %22 = ADORA.BlockLoad %arg1 [%20, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %23 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %24 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %25 = affine.apply #map2(%arg3)
      ADORA.BlockStore %10, %arg0 [%25, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %26 = ADORA.BlockLoad %arg0 [%25, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %27 = ADORA.BlockLoad %arg1 [%25, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %28 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %29 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      ADORA.kernel {
        affine.for %arg4 = 0 to 25 {
          %33 = affine.load %11[0, %arg4] : memref<1x25xf32>
          %34 = affine.apply #map(%arg3)
          %35 = affine.load %16[0, %arg4] : memref<1x25xf32>
          %36 = affine.apply #map1(%arg3)
          %37 = affine.load %21[0, %arg4] : memref<1x25xf32>
          %38 = affine.apply #map2(%arg3)
          %39 = affine.load %26[0, %arg4] : memref<1x25xf32>
          %40:4 = affine.for %arg5 = 0 to 30 iter_args(%arg6 = %33, %arg7 = %35, %arg8 = %37, %arg9 = %39) -> (f32, f32, f32, f32) {
            %44 = affine.load %12[0, %arg5] : memref<1x30xf32>
            %45 = arith.mulf %44, %cst_0 : f32
            %46 = affine.load %13[%arg5, %arg4] : memref<30x25xf32>
            %47 = arith.mulf %45, %46 : f32
            %48 = arith.addf %arg6, %47 : f32
            %49 = affine.apply #map(%arg3)
            %50 = affine.load %17[0, %arg5] : memref<1x30xf32>
            %51 = arith.mulf %50, %cst_0 : f32
            %52 = affine.load %18[%arg5, %arg4] : memref<30x25xf32>
            %53 = arith.mulf %51, %52 : f32
            %54 = arith.addf %arg7, %53 : f32
            %55 = affine.apply #map1(%arg3)
            %56 = affine.load %22[0, %arg5] : memref<1x30xf32>
            %57 = arith.mulf %56, %cst_0 : f32
            %58 = affine.load %23[%arg5, %arg4] : memref<30x25xf32>
            %59 = arith.mulf %57, %58 : f32
            %60 = arith.addf %arg8, %59 : f32
            %61 = affine.apply #map2(%arg3)
            %62 = affine.load %27[0, %arg5] : memref<1x30xf32>
            %63 = arith.mulf %62, %cst_0 : f32
            %64 = affine.load %28[%arg5, %arg4] : memref<30x25xf32>
            %65 = arith.mulf %63, %64 : f32
            %66 = arith.addf %arg9, %65 : f32
            affine.yield %48, %54, %60, %66 : f32, f32, f32, f32
          }
          affine.store %40#0, %14[0, %arg4] : memref<1x25xf32>
          %41 = affine.apply #map(%arg3)
          affine.store %40#1, %19[0, %arg4] : memref<1x25xf32>
          %42 = affine.apply #map1(%arg3)
          affine.store %40#2, %24[0, %arg4] : memref<1x25xf32>
          %43 = affine.apply #map2(%arg3)
          affine.store %40#3, %29[0, %arg4] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_gemm_1"}
      ADORA.BlockStore %14, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %30 = affine.apply #map(%arg3)
      ADORA.BlockStore %19, %arg0 [%30, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %31 = affine.apply #map1(%arg3)
      ADORA.BlockStore %24, %arg0 [%31, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %32 = affine.apply #map2(%arg3)
      ADORA.BlockStore %29, %arg0 [%32, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
    }
    return
  }
}
