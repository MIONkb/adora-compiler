module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @merge_MATMUL_4x4(%arg0: memref<?x36xi32>, %arg1: memref<?x36xi32>, %arg2: memref<?x36xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x36xi32> -> memref<33x36xi32>  {Id = "0", KernelName = "merge_MATMUL_4x4"}
    %1 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x36xi32> -> memref<36x33xi32>  {Id = "1", KernelName = "merge_MATMUL_4x4"}
    %2 = ADORA.BlockLoad %arg1 [0, 1] : memref<?x36xi32> -> memref<36x33xi32>  {Id = "2", KernelName = "merge_MATMUL_4x4"}
    %3 = ADORA.BlockLoad %arg1 [0, 2] : memref<?x36xi32> -> memref<36x33xi32>  {Id = "3", KernelName = "merge_MATMUL_4x4"}
    %4 = ADORA.BlockLoad %arg1 [0, 3] : memref<?x36xi32> -> memref<36x33xi32>  {Id = "4", KernelName = "merge_MATMUL_4x4"}
    %5 = ADORA.BlockLoad %arg0 [1, 0] : memref<?x36xi32> -> memref<33x36xi32>  {Id = "5", KernelName = "merge_MATMUL_4x4"}
    %6 = ADORA.BlockLoad %arg0 [2, 0] : memref<?x36xi32> -> memref<33x36xi32>  {Id = "6", KernelName = "merge_MATMUL_4x4"}
    %7 = ADORA.BlockLoad %arg0 [3, 0] : memref<?x36xi32> -> memref<33x36xi32>  {Id = "7", KernelName = "merge_MATMUL_4x4"}
    %80 = ADORA.LocalMemAlloc memref<9x36xi32>  {Id = "8", KernelName = "merge_MATMUL_4x4"}
    %81 = ADORA.LocalMemAlloc memref<9x36xi32>  {Id = "9", KernelName = "merge_MATMUL_4x4"}
    %82 = ADORA.LocalMemAlloc memref<9x36xi32>  {Id = "10", KernelName = "merge_MATMUL_4x4"}
    %83 = ADORA.LocalMemAlloc memref<9x36xi32>  {Id = "11", KernelName = "merge_MATMUL_4x4"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 9 {
        affine.for %arg4 = 0 to 9 {
          %9:16 = affine.for %arg5 = 0 to 36 iter_args(%arg6 = %c0_i32, %arg7 = %c0_i32, %arg8 = %c0_i32, %arg9 = %c0_i32, %arg10 = %c0_i32, %arg11 = %c0_i32, %arg12 = %c0_i32, %arg13 = %c0_i32, %arg14 = %c0_i32, %arg15 = %c0_i32, %arg16 = %c0_i32, %arg17 = %c0_i32, %arg18 = %c0_i32, %arg19 = %c0_i32, %arg20 = %c0_i32, %arg21 = %c0_i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
            %10 = affine.load %0[%arg3 * 4, %arg5] : memref<33x36xi32>
            %11 = affine.load %1[%arg5, %arg4 * 4] : memref<36x33xi32>
            %12 = arith.muli %10, %11 : i32
            %13 = arith.addi %arg6, %12 : i32
            %14 = affine.load %2[%arg5, %arg4 * 4] : memref<36x33xi32>
            %15 = arith.muli %10, %14 : i32
            %16 = arith.addi %arg7, %15 : i32
            %17 = affine.load %3[%arg5, %arg4 * 4] : memref<36x33xi32>
            %18 = arith.muli %10, %17 : i32
            %19 = arith.addi %arg8, %18 : i32
            %20 = affine.load %4[%arg5, %arg4 * 4] : memref<36x33xi32>
            %21 = arith.muli %10, %20 : i32
            %22 = arith.addi %arg9, %21 : i32
            %23 = affine.load %5[%arg3 * 4, %arg5] : memref<33x36xi32>
            %24 = arith.muli %23, %11 : i32
            %25 = arith.addi %arg10, %24 : i32
            %26 = arith.muli %23, %14 : i32
            %27 = arith.addi %arg11, %26 : i32
            %28 = arith.muli %23, %17 : i32
            %29 = arith.addi %arg12, %28 : i32
            %30 = arith.muli %23, %20 : i32
            %31 = arith.addi %arg13, %30 : i32
            %32 = affine.load %6[%arg3 * 4, %arg5] : memref<33x36xi32>
            %33 = arith.muli %32, %11 : i32
            %34 = arith.addi %arg14, %33 : i32
            %35 = arith.muli %32, %14 : i32
            %36 = arith.addi %arg15, %35 : i32
            %37 = arith.muli %32, %17 : i32
            %38 = arith.addi %arg16, %37 : i32
            %39 = arith.muli %32, %20 : i32
            %40 = arith.addi %arg17, %39 : i32
            %41 = affine.load %7[%arg3 * 4, %arg5] : memref<33x36xi32>
            %42 = arith.muli %41, %11 : i32
            %43 = arith.addi %arg18, %42 : i32
            %44 = arith.muli %41, %14 : i32
            %45 = arith.addi %arg19, %44 : i32
            %46 = arith.muli %41, %17 : i32
            %47 = arith.addi %arg20, %46 : i32
            %48 = arith.muli %41, %20 : i32
            %49 = arith.addi %arg21, %48 : i32
            affine.yield %13, %16, %19, %22, %25, %27, %29, %31, %34, %36, %38, %40, %43, %45, %47, %49 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
          }
          %merge0 = ADORA.merge %9#0, %9#1, %9#2, %9#3 : i32, i32, i32, i32 -> vector<4xi32>
          affine.vector_store %merge0, %80[%arg3, %arg4 * 4] : memref<9x36xi32>, vector<4xi32>
          
          %merge1 = ADORA.merge %9#4, %9#5, %9#6, %9#7 : i32, i32, i32, i32 -> vector<4xi32>
          affine.vector_store %merge1, %81[%arg3, %arg4 * 4]: memref<9x36xi32>, vector<4xi32>
          
          %merge2 = ADORA.merge %9#8, %9#9, %9#10, %9#11 : i32, i32, i32, i32 -> vector<4xi32>
          affine.vector_store %merge2, %82[%arg3, %arg4 * 4] : memref<9x36xi32>, vector<4xi32>
          
          %merge3 = ADORA.merge %9#12, %9#13, %9#14, %9#15 : i32, i32, i32, i32 -> vector<4xi32>
          affine.vector_store %merge3, %83[%arg3, %arg4 * 4] : memref<9x36xi32>, vector<4xi32>

          // affine.store %9#15, %8[%arg3 * 4 + 3, %arg4 * 4 + 3] : memref<36x36xi32>
          // affine.store %9#14, %8[%arg3 * 4 + 3, %arg4 * 4 + 2] : memref<36x36xi32>
          // affine.store %9#13, %8[%arg3 * 4 + 3, %arg4 * 4 + 1] : memref<36x36xi32>
          // affine.store %9#12, %8[%arg3 * 4 + 3, %arg4 * 4] : memref<36x36xi32>
          // affine.store %9#11, %8[%arg3 * 4 + 2, %arg4 * 4 + 3] : memref<36x36xi32>
          // affine.store %9#10, %8[%arg3 * 4 + 2, %arg4 * 4 + 2] : memref<36x36xi32>
          // affine.store %9#9, %8[%arg3 * 4 + 2, %arg4 * 4 + 1] : memref<36x36xi32>
          // affine.store %9#8, %8[%arg3 * 4 + 2, %arg4 * 4] : memref<36x36xi32>
          // affine.store %9#7, %8[%arg3 * 4 + 1, %arg4 * 4 + 3] : memref<36x36xi32>
          // affine.store %9#6, %8[%arg3 * 4 + 1, %arg4 * 4 + 2] : memref<36x36xi32>
          // affine.store %9#5, %8[%arg3 * 4 + 1, %arg4 * 4 + 1] : memref<36x36xi32>
          // affine.store %9#4, %8[%arg3 * 4 + 1, %arg4 * 4] : memref<36x36xi32>
          // affine.store %9#3, %8[%arg3 * 4, %arg4 * 4 + 3] : memref<36x36xi32>
          // affine.store %9#2, %8[%arg3 * 4, %arg4 * 4 + 2] : memref<36x36xi32>
          // affine.store %9#1, %8[%arg3 * 4, %arg4 * 4 + 1] : memref<36x36xi32>
          // affine.store %9#0, %8[%arg3 * 4, %arg4 * 4] : memref<36x36xi32>

        }
      }
      ADORA.terminator
    } {KernelName = "merge_MATMUL_4x4"}

    // It is wrong here.
    ADORA.BlockStore %80, %arg2 [0, 0] : memref<9x36xi32> -> memref<?x36xi32>  {Id = "8", KernelName = "merge_MATMUL_4x4"}
    ADORA.BlockStore %81, %arg2 [9, 0] : memref<9x36xi32> -> memref<?x36xi32>  {Id = "9", KernelName = "merge_MATMUL_4x4"}
    ADORA.BlockStore %82, %arg2 [18, 0] : memref<9x36xi32> -> memref<?x36xi32>  {Id = "10", KernelName = "merge_MATMUL_4x4"}
    ADORA.BlockStore %83, %arg2 [27, 0] : memref<9x36xi32> -> memref<?x36xi32>  {Id = "11", KernelName = "merge_MATMUL_4x4"}
    return
  }
}

