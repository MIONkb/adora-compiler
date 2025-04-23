#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.333300e-01 : f32
    affine.for %arg2 = 0 to 500 {
      %0 = ADORA.BlockLoad %arg0 [0] : memref<2000xf32> -> memref<2000xf32>  {Id = "0", KernelName = "kernel_jacobi_1d_0"}
      %1 = ADORA.BlockLoad %arg0 [1] : memref<2000xf32> -> memref<2000xf32>  {Id = "1", KernelName = "kernel_jacobi_1d_0"}
      %2 = ADORA.BlockLoad %arg0 [2] : memref<2000xf32> -> memref<2000xf32>  {Id = "2", KernelName = "kernel_jacobi_1d_0"}
      %3 = ADORA.LocalMemAlloc memref<2000xf32>  {Id = "3", KernelName = "kernel_jacobi_1d_0"}
      ADORA.kernel {
        affine.for %arg3 = 0 to 1998 {
          %8 = affine.apply #map(%arg3)
          %9 = affine.load %0[%8 - 1] : memref<2000xf32>
          %10 = affine.load %1[%8 - 1] : memref<2000xf32>
          %11 = arith.addf %9, %10 : f32
          %12 = affine.load %2[%8 - 1] : memref<2000xf32>
          %13 = arith.addf %11, %12 : f32
          %14 = arith.mulf %13, %cst : f32
          affine.store %14, %3[%8 - 1] : memref<2000xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_jacobi_1d_0"}
      ADORA.BlockStore %3, %arg1 [1] : memref<2000xf32> -> memref<2000xf32>  {Id = "3", KernelName = "kernel_jacobi_1d_0"}
      %4 = ADORA.BlockLoad %arg1 [0] : memref<2000xf32> -> memref<2000xf32>  {Id = "0", KernelName = "kernel_jacobi_1d_1"}
      %5 = ADORA.BlockLoad %arg1 [1] : memref<2000xf32> -> memref<2000xf32>  {Id = "1", KernelName = "kernel_jacobi_1d_1"}
      %6 = ADORA.BlockLoad %arg1 [2] : memref<2000xf32> -> memref<2000xf32>  {Id = "2", KernelName = "kernel_jacobi_1d_1"}
      %7 = ADORA.LocalMemAlloc memref<2000xf32>  {Id = "3", KernelName = "kernel_jacobi_1d_1"}
      ADORA.kernel {
        affine.for %arg3 = 0 to 1998 {
          %8 = affine.apply #map(%arg3)
          %9 = affine.load %4[%8 - 1] : memref<2000xf32>
          %10 = affine.load %5[%8 - 1] : memref<2000xf32>
          %11 = arith.addf %9, %10 : f32
          %12 = affine.load %6[%8 - 1] : memref<2000xf32>
          %13 = arith.addf %11, %12 : f32
          %14 = arith.mulf %13, %cst : f32
          affine.store %14, %7[%8 - 1] : memref<2000xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_jacobi_1d_1"}
      ADORA.BlockStore %7, %arg0 [1] : memref<2000xf32> -> memref<2000xf32>  {Id = "3", KernelName = "kernel_jacobi_1d_1"}
    }
    return
  }
}

