module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @merge_MATMUL(%arg0: memref<?x4xi32>, %arg1: memref<?x4xi32>, %arg2: memref<?x4xi32>, %arg3: memref<?x4xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "0", KernelName = "merge_MATMUL"}
    %1 = ADORA.BlockLoad %arg2 [0, 1] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "1", KernelName = "merge_MATMUL"}
    %2 = ADORA.BlockLoad %arg2 [0, 2] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "2", KernelName = "merge_MATMUL"}
    %3 = ADORA.BlockLoad %arg2 [0, 3] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "3", KernelName = "merge_MATMUL"}
    %4 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x4xi32> -> memref<4x4xi32>  {Id = "4", KernelName = "merge_MATMUL"}
    %5 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "5", KernelName = "merge_MATMUL"}
    %6 = ADORA.BlockLoad %arg1 [0, 1] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "6", KernelName = "merge_MATMUL"}
    %7 = ADORA.BlockLoad %arg1 [0, 2] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "7", KernelName = "merge_MATMUL"}
    %8 = ADORA.BlockLoad %arg1 [0, 3] : memref<?x4xi32> -> memref<4x2xi32>  {Id = "8", KernelName = "merge_MATMUL"}
    %9 = ADORA.LocalMemAlloc memref<4x4xi32>  {Id = "9", KernelName = "merge_MATMUL"}
    ADORA.kernel {
      affine.for %arg4 = 0 to 4 {
        %10 = affine.load %0[%arg4, 0] : memref<4x2xi32>
        %11 = affine.load %1[%arg4, 0] : memref<4x2xi32>
        %12 = affine.load %2[%arg4, 0] : memref<4x2xi32>
        %13 = affine.load %3[%arg4, 0] : memref<4x2xi32>
        %14:4 = affine.for %arg5 = 0 to 4 iter_args(%arg6 = %10, %arg7 = %11, %arg8 = %12, %arg9 = %13) -> (i32, i32, i32, i32) {
          %15 = affine.load %4[%arg4, %arg5] : memref<4x4xi32>
          %16 = affine.load %5[%arg5, 0] : memref<4x2xi32>
          %17 = arith.muli %15, %16 : i32
          %18 = arith.addi %arg6, %17 : i32
          %19 = affine.load %6[%arg5, 0] : memref<4x2xi32>
          %20 = arith.muli %15, %19 : i32
          %21 = arith.addi %arg7, %20 : i32
          %22 = affine.load %7[%arg5, 0] : memref<4x2xi32>
          %23 = arith.muli %15, %22 : i32
          %24 = arith.addi %arg8, %23 : i32
          %25 = affine.load %8[%arg5, 0] : memref<4x2xi32>
          %26 = arith.muli %15, %25 : i32
          %27 = arith.addi %arg9, %26 : i32
          affine.yield %18, %21, %24, %27 : i32, i32, i32, i32
        }
        %merge = ADORA.merge %14#0, %14#1, %14#2, %14#3 : i32, i32, i32, i32 -> vector<4xi32>
        affine.vector_store %merge, %9[%arg4, 0] : memref<4x4xi32>, vector<4xi32>
        // affine.for %arg7 = 0 to 4 {
        //   %elem = vector.extract %merge[%arg7] : vector<4xi32>
        //   // affine.vector_store %elem, %arg4[%arg5 * 4] : memref<?xi32>, vector<4xi32>
        //   affine.store %elem, %9[%arg4, %arg7] : memref<4x4xi32>
        // }
        // affine.store %14#3, %9[%arg4, 3] : memref<4x4xi32>
        // affine.store %14#2, %9[%arg4, 2] : memref<4x4xi32>
        // affine.store %14#1, %9[%arg4, 1] : memref<4x4xi32>
        // affine.store %14#0, %9[%arg4, 0] : memref<4x4xi32>
      }
      ADORA.terminator
    } {KernelName = "merge_MATMUL"}
    ADORA.BlockStore %9, %arg3 [0, 0] : memref<4x4xi32> -> memref<?x4xi32>  {Id = "9", KernelName = "merge_MATMUL"}
    return
  }
}

