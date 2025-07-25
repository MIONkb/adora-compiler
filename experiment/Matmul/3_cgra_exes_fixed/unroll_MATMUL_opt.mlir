module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @unroll_MATMUL(%arg0: memref<?x4xi32>, %arg1: memref<?x4xi32>, %arg2: memref<?x4xi32>, %arg3: memref<?x4xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x4xi32> -> memref<4x4xi32>  {Id = "0", KernelName = "unroll_MATMUL"}
    %1 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x4xi32> -> memref<4x4xi32>  {Id = "1", KernelName = "unroll_MATMUL"}
    %2 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x4xi32> -> memref<4x4xi32>  {Id = "2", KernelName = "unroll_MATMUL"}
    %3 = ADORA.LocalMemAlloc memref<4x4xi32>  {Id = "3", KernelName = "unroll_MATMUL"}
    ADORA.kernel {
      affine.for %arg4 = 0 to 4 {
        affine.for %arg5 = 0 to 4 {
          %4 = affine.load %0[%arg4, %arg5] : memref<4x4xi32>
          %5 = affine.for %arg6 = 0 to 4 iter_args(%arg7 = %4) -> (i32) {
            %6 = affine.load %1[%arg4, %arg6] : memref<4x4xi32>
            %7 = affine.load %2[%arg6, %arg5] : memref<4x4xi32>
            %8 = arith.muli %6, %7 : i32
            %9 = arith.addi %arg7, %8 : i32
            affine.yield %9 : i32
          }
          affine.store %5, %3[%arg4, %arg5] : memref<4x4xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "unroll_MATMUL"}
    ADORA.BlockStore %3, %arg3 [0, 0] : memref<4x4xi32> -> memref<?x4xi32>  {Id = "3", KernelName = "unroll_MATMUL"}
    return
  }
}

