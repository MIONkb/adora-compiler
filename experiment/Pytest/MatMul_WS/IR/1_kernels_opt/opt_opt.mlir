module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @merge_MATMUL_4x4_IS(%arg0: memref<?x36xi32>, %arg1: memref<?x36xi32>, %arg2: memref<?x36xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x36xi32> -> memref<36x10xi32>  {Id = "0", KernelName = "merge_MATMUL_4x4_IS"}
    %1 = ADORA.BlockLoad %0 [0, 0] : memref<36x10xi32> -> memref<36x10xi32>  {Id = "1", KernelName = "merge_MATMUL_4x4_IS"}
    %2 = ADORA.BlockLoad %1 [0, 0] : memref<36x10xi32> -> memref<36x10xi32>  {Id = "2", KernelName = "merge_MATMUL_4x4_IS"}
    %3 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x36xi32> -> memref<36x10xi32>  {Id = "3", KernelName = "merge_MATMUL_4x4_IS"}
    %4 = ADORA.BlockLoad %3 [0, 0] : memref<36x10xi32> -> memref<36x10xi32>  {Id = "4", KernelName = "merge_MATMUL_4x4_IS"}
    %5 = ADORA.BlockLoad %4 [0, 0] : memref<36x10xi32> -> memref<36x10xi32>  {Id = "5", KernelName = "merge_MATMUL_4x4_IS"}
    %6 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x36xi32> -> memref<36x36xi32>  {Id = "6", KernelName = "merge_MATMUL_4x4_IS"}
    %7 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x36xi32> -> memref<36x9xi32>  {Id = "7", KernelName = "merge_MATMUL_4x4_IS"}
    %8 = ADORA.BlockLoad %arg1 [0, 1] : memref<?x36xi32> -> memref<36x9xi32>  {Id = "8", KernelName = "merge_MATMUL_4x4_IS"}
    %9 = ADORA.BlockLoad %arg1 [0, 2] : memref<?x36xi32> -> memref<36x9xi32>  {Id = "9", KernelName = "merge_MATMUL_4x4_IS"}
    %10 = ADORA.BlockLoad %arg1 [0, 3] : memref<?x36xi32> -> memref<36x9xi32>  {Id = "10", KernelName = "merge_MATMUL_4x4_IS"}
    %11 = ADORA.BlockLoad %arg2 [0, 2] : memref<?x36xi32> -> memref<36x10xi32>  {Id = "0", KernelName = "merge_MATMUL_4x4_IS"}
    %12 = ADORA.BlockLoad %arg2 [0, 2] : memref<?x36xi32> -> memref<36x10xi32>  {Id = "1", KernelName = "merge_MATMUL_4x4_IS"}
    %13 = ADORA.BlockLoad %6 [0, 0] : memref<36x36xi32> -> memref<36x36xi32>  {Id = "2", KernelName = "merge_MATMUL_4x4_IS"}
    %14 = ADORA.BlockLoad %7 [0, 0] : memref<36x9xi32> -> memref<36x9xi32>  {Id = "3", KernelName = "merge_MATMUL_4x4_IS"}
    %15 = ADORA.BlockLoad %5 [0, 0] : memref<36x10xi32> -> memref<36x10xi32>  {Id = "4", KernelName = "merge_MATMUL_4x4_IS"}
    %16 = ADORA.BlockLoad %0 [0, 1] : memref<36x10xi32> -> memref<36x10xi32>  {Id = "5", KernelName = "merge_MATMUL_4x4_IS"}
    %17 = ADORA.LocalMemAlloc memref<36x10xi32>  {Id = "6", KernelName = "merge_MATMUL_4x4_IS"}
    %18 = ADORA.LocalMemAlloc memref<36x10xi32>  {Id = "7", KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 9 {
        affine.for %arg4 = 0 to 36 {
          affine.for %arg5 = 0 to 36 {
            %19 = affine.load %13[%arg5, %arg4] : memref<36x36xi32>
            %20 = affine.load %14[%arg4, %arg3] : memref<36x9xi32>
            %21 = arith.muli %19, %20 : i32
            %22 = affine.load %15[%arg5, %arg3] : memref<36x10xi32>
            %23 = arith.addi %22, %21 : i32
            affine.store %23, %18[%arg5, %arg3] : memref<36x10xi32>
            %24 = arith.muli %19, %20 : i32
            %25 = affine.load %16[%arg5, %arg3] : memref<36x10xi32>
            %26 = arith.addi %25, %24 : i32
            affine.store %26, %17[%arg5, %arg3] : memref<36x10xi32>
            %27 = arith.muli %19, %20 : i32
            %28 = affine.load %12[%arg5, %arg3] : memref<36x10xi32>
            %29 = arith.addi %28, %27 : i32
            affine.store %29, %11[%arg5, %arg3] : memref<36x10xi32>
            %30 = arith.muli %19, %20 : i32
            %31 = affine.load %11[%arg5, %arg3 + 1] : memref<36x10xi32>
            %32 = arith.addi %31, %30 : i32
            affine.store %32, %12[%arg5, %arg3 + 1] : memref<36x10xi32>
          }
        }
      }
      ADORA.terminator
    } {KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.BlockStore %18, %2 [0, 0] : memref<36x10xi32> -> memref<36x10xi32>  {Id = "7", KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.BlockStore %17, %3 [0, 1] : memref<36x10xi32> -> memref<36x10xi32>  {Id = "6", KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.BlockStore %12, %arg2 [0, 2] : memref<36x10xi32> -> memref<?x36xi32>  {Id = "1", KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.BlockStore %11, %arg2 [0, 2] : memref<36x10xi32> -> memref<?x36xi32>  {Id = "0", KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.BlockStore %5, %4 [0, 0] : memref<36x10xi32> -> memref<36x10xi32>  {Id = "5", KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.BlockStore %4, %3 [0, 0] : memref<36x10xi32> -> memref<36x10xi32>  {Id = "4", KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.BlockStore %3, %arg2 [0, 0] : memref<36x10xi32> -> memref<?x36xi32>  {Id = "3", KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.BlockStore %2, %1 [0, 0] : memref<36x10xi32> -> memref<36x10xi32>  {Id = "2", KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.BlockStore %1, %0 [0, 0] : memref<36x10xi32> -> memref<36x10xi32>  {Id = "1", KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.BlockStore %0, %arg2 [0, 0] : memref<36x10xi32> -> memref<?x36xi32>  {Id = "0", KernelName = "merge_MATMUL_4x4_IS"}
    return
  }
}

