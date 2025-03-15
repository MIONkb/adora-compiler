module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_mvt(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x40xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = ADORA.BlockLoad %arg0 [0] : memref<?xf32> -> memref<40xf32>  {Id = "0", KernelName = "kernel_mvt_0"}
    %1 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x40xf32> -> memref<40x40xf32>  {Id = "1", KernelName = "kernel_mvt_0"}
    %2 = ADORA.BlockLoad %arg2 [0] : memref<?xf32> -> memref<40xf32>  {Id = "2", KernelName = "kernel_mvt_0"}
    %3 = ADORA.LocalMemAlloc memref<40xf32>  {Id = "3", KernelName = "kernel_mvt_0"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 40 {
        %8 = affine.load %0[%arg5] : memref<40xf32>
        %9 = affine.for %arg6 = 0 to 40 iter_args(%arg7 = %8) -> (f32) {
          %10 = affine.load %1[%arg5, %arg6] : memref<40x40xf32>
          %11 = affine.load %2[%arg6] : memref<40xf32>
          %12 = arith.mulf %10, %11 : f32
          %13 = arith.addf %arg7, %12 : f32
          affine.yield %13 : f32
        }
        affine.store %9, %3[%arg5] : memref<40xf32>
      }
      ADORA.terminator
    } {KernelName = "kernel_mvt_0"}
    ADORA.BlockStore %3, %arg0 [0] : memref<40xf32> -> memref<?xf32>  {Id = "3", KernelName = "kernel_mvt_0"}
    %4 = ADORA.BlockLoad %arg1 [0] : memref<?xf32> -> memref<40xf32>  {Id = "0", KernelName = "kernel_mvt_1"}
    %5 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x40xf32> -> memref<40x40xf32>  {Id = "1", KernelName = "kernel_mvt_1"}
    %6 = ADORA.BlockLoad %arg3 [0] : memref<?xf32> -> memref<40xf32>  {Id = "2", KernelName = "kernel_mvt_1"}
    %7 = ADORA.LocalMemAlloc memref<40xf32>  {Id = "3", KernelName = "kernel_mvt_1"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 40 {
        %8 = affine.load %4[%arg5] : memref<40xf32>
        %9 = affine.for %arg6 = 0 to 40 iter_args(%arg7 = %8) -> (f32) {
          %10 = affine.load %5[%arg6, %arg5] : memref<40x40xf32>
          %11 = affine.load %6[%arg6] : memref<40xf32>
          %12 = arith.mulf %10, %11 : f32
          %13 = arith.addf %arg7, %12 : f32
          affine.yield %13 : f32
        }
        affine.store %9, %7[%arg5] : memref<40xf32>
      }
      ADORA.terminator
    } {KernelName = "kernel_mvt_1"}
    ADORA.BlockStore %7, %arg1 [0] : memref<40xf32> -> memref<?xf32>  {Id = "3", KernelName = "kernel_mvt_1"}
    return
  }
}

