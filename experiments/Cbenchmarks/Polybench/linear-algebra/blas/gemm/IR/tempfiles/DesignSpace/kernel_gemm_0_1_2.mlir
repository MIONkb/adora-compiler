#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm(%arg0: memref<?x25xf32>, %arg1: memref<?x30xf32>, %arg2: memref<?x25xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.200000e+00 : f32
    %cst_0 = arith.constant 1.500000e+00 : f32
    affine.for %arg3 = 0 to 20 step 2 {
      %0 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
      %1 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %2 = affine.apply #map(%arg3)
      %3 = ADORA.BlockLoad %arg0 [%2, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
      %4 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      ADORA.kernel {
        affine.for %arg4 = 0 to 25 {
          %15 = affine.load %0[0, %arg4] : memref<1x25xf32>
          %16 = arith.mulf %15, %cst : f32
          affine.store %16, %1[0, %arg4] : memref<1x25xf32>
          %17 = affine.load %3[0, %arg4] : memref<1x25xf32>
          %18 = arith.mulf %17, %cst : f32
          affine.store %18, %4[0, %arg4] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_gemm_0"}
      ADORA.BlockStore %1, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %5 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %6 = ADORA.BlockLoad %arg1 [%arg3, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %7 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %8 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %9 = affine.apply #map(%arg3)
      ADORA.BlockStore %4, %arg0 [%9, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %10 = ADORA.BlockLoad %arg0 [%9, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %11 = ADORA.BlockLoad %arg1 [%9, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %12 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %13 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      ADORA.kernel {
        affine.for %arg4 = 0 to 25 {
          %15 = affine.load %5[0, %arg4] : memref<1x25xf32>
          %16 = affine.apply #map(%arg3)
          %17 = affine.load %10[0, %arg4] : memref<1x25xf32>
          %18:2 = affine.for %arg5 = 0 to 30 iter_args(%arg6 = %15, %arg7 = %17) -> (f32, f32) {
            %20 = affine.load %6[0, %arg5] : memref<1x30xf32>
            %21 = arith.mulf %20, %cst_0 : f32
            %22 = affine.load %7[%arg5, %arg4] : memref<30x25xf32>
            %23 = arith.mulf %21, %22 : f32
            %24 = arith.addf %arg6, %23 : f32
            %25 = affine.apply #map(%arg3)
            %26 = affine.load %11[0, %arg5] : memref<1x30xf32>
            %27 = arith.mulf %26, %cst_0 : f32
            %28 = affine.load %12[%arg5, %arg4] : memref<30x25xf32>
            %29 = arith.mulf %27, %28 : f32
            %30 = arith.addf %arg7, %29 : f32
            affine.yield %24, %30 : f32, f32
          }
          affine.store %18#0, %8[0, %arg4] : memref<1x25xf32>
          %19 = affine.apply #map(%arg3)
          affine.store %18#1, %13[0, %arg4] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_gemm_1"}
      ADORA.BlockStore %8, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      %14 = affine.apply #map(%arg3)
      ADORA.BlockStore %13, %arg0 [%14, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
    }
    return
  }
}
