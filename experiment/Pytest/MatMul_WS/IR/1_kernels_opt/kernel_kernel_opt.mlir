module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @merge_MATMUL_4x4_IS(%arg0: memref<?x36xi32>, %arg1: memref<?x36xi32>, %arg2: memref<?x36xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x36xi32> -> memref<36x9xi32>  {Id = "0", KernelName = "merge_MATMUL_4x4_IS"}
    %1 = ADORA.BlockLoad %arg2 [0, 2] : memref<?x36xi32> -> memref<36x9xi32>  {Id = "1", KernelName = "merge_MATMUL_4x4_IS"}
    %2 = ADORA.BlockLoad %arg2 [0, 1] : memref<?x36xi32> -> memref<36x9xi32>  {Id = "2", KernelName = "merge_MATMUL_4x4_IS"}
    %3 = ADORA.BlockLoad %arg2 [0, 3] : memref<?x36xi32> -> memref<36x9xi32>  {Id = "3", KernelName = "merge_MATMUL_4x4_IS"}
    %4 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x36xi32> -> memref<36x9xi32>  {Id = "4", KernelName = "merge_MATMUL_4x4_IS"}
    %5 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "5", KernelName = "merge_MATMUL_4x4_IS"}
    %6 = ADORA.BlockLoad %arg0 [0, 1] : memref<?x36xi32> -> memref<36x9xi32>  {Id = "6", KernelName = "merge_MATMUL_4x4_IS"}
    %7 = ADORA.BlockLoad %arg1 [1, 0] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "7", KernelName = "merge_MATMUL_4x4_IS"}
    %8 = ADORA.BlockLoad %arg0 [0, 2] : memref<?x36xi32> -> memref<36x9xi32>  {Id = "8", KernelName = "merge_MATMUL_4x4_IS"}
    %9 = ADORA.BlockLoad %arg1 [2, 0] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "9", KernelName = "merge_MATMUL_4x4_IS"}
    %10 = ADORA.BlockLoad %arg0 [0, 3] : memref<?x36xi32> -> memref<36x9xi32>  {Id = "10", KernelName = "merge_MATMUL_4x4_IS"}
    %11 = ADORA.BlockLoad %arg1 [3, 0] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "11", KernelName = "merge_MATMUL_4x4_IS"}
    %12 = ADORA.BlockLoad %arg1 [0, 1] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "12", KernelName = "merge_MATMUL_4x4_IS"}
    %13 = ADORA.BlockLoad %arg1 [1, 1] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "13", KernelName = "merge_MATMUL_4x4_IS"}
    %14 = ADORA.BlockLoad %arg1 [2, 1] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "14", KernelName = "merge_MATMUL_4x4_IS"}
    %15 = ADORA.BlockLoad %arg1 [3, 1] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "15", KernelName = "merge_MATMUL_4x4_IS"}
    %16 = ADORA.BlockLoad %arg1 [0, 2] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "16", KernelName = "merge_MATMUL_4x4_IS"}
    %17 = ADORA.BlockLoad %arg1 [1, 2] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "17", KernelName = "merge_MATMUL_4x4_IS"}
    %18 = ADORA.BlockLoad %arg1 [2, 2] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "18", KernelName = "merge_MATMUL_4x4_IS"}
    %19 = ADORA.BlockLoad %arg1 [3, 2] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "19", KernelName = "merge_MATMUL_4x4_IS"}
    %20 = ADORA.BlockLoad %arg1 [0, 3] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "20", KernelName = "merge_MATMUL_4x4_IS"}
    %21 = ADORA.BlockLoad %arg1 [1, 3] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "21", KernelName = "merge_MATMUL_4x4_IS"}
    %22 = ADORA.BlockLoad %arg1 [2, 3] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "22", KernelName = "merge_MATMUL_4x4_IS"}
    %23 = ADORA.BlockLoad %arg1 [3, 3] : memref<?x36xi32> -> memref<9x9xi32>  {Id = "23", KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 9 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 36 {
            %24 = affine.load %4[%arg5, %arg4] : memref<36x9xi32>
            %25 = affine.load %5[%arg4, %arg3] : memref<9x9xi32>
            %26 = arith.muli %24, %25 : i32
            %27 = affine.load %0[%arg5, %arg3] : memref<36x9xi32>
            %28 = arith.addi %27, %26 : i32
            %29 = affine.load %6[%arg5, %arg4] : memref<36x9xi32>
            %30 = affine.load %7[%arg4, %arg3] : memref<9x9xi32>
            %31 = arith.muli %29, %30 : i32
            %32 = arith.addi %28, %31 : i32
            %33 = affine.load %8[%arg5, %arg4] : memref<36x9xi32>
            %34 = affine.load %9[%arg4, %arg3] : memref<9x9xi32>
            %35 = arith.muli %33, %34 : i32
            %36 = arith.addi %32, %35 : i32
            %37 = affine.load %10[%arg5, %arg4] : memref<36x9xi32>
            %38 = affine.load %11[%arg4, %arg3] : memref<9x9xi32>
            %39 = arith.muli %37, %38 : i32
            %40 = arith.addi %36, %39 : i32
            affine.store %40, %0[%arg5, %arg3] : memref<36x9xi32>
            %41 = affine.load %12[%arg4, %arg3] : memref<9x9xi32>
            %42 = arith.muli %24, %41 : i32
            %43 = affine.load %2[%arg5, %arg3] : memref<36x9xi32>
            %44 = arith.addi %43, %42 : i32
            %45 = affine.load %13[%arg4, %arg3] : memref<9x9xi32>
            %46 = arith.muli %29, %45 : i32
            %47 = arith.addi %44, %46 : i32
            %48 = affine.load %14[%arg4, %arg3] : memref<9x9xi32>
            %49 = arith.muli %33, %48 : i32
            %50 = arith.addi %47, %49 : i32
            %51 = affine.load %15[%arg4, %arg3] : memref<9x9xi32>
            %52 = arith.muli %37, %51 : i32
            %53 = arith.addi %50, %52 : i32
            affine.store %53, %2[%arg5, %arg3] : memref<36x9xi32>
            %54 = affine.load %16[%arg4, %arg3] : memref<9x9xi32>
            %55 = arith.muli %24, %54 : i32
            %56 = affine.load %1[%arg5, %arg3] : memref<36x9xi32>
            %57 = arith.addi %56, %55 : i32
            %58 = affine.load %17[%arg4, %arg3] : memref<9x9xi32>
            %59 = arith.muli %29, %58 : i32
            %60 = arith.addi %57, %59 : i32
            %61 = affine.load %18[%arg4, %arg3] : memref<9x9xi32>
            %62 = arith.muli %33, %61 : i32
            %63 = arith.addi %60, %62 : i32
            %64 = affine.load %19[%arg4, %arg3] : memref<9x9xi32>
            %65 = arith.muli %37, %64 : i32
            %66 = arith.addi %63, %65 : i32
            affine.store %66, %1[%arg5, %arg3] : memref<36x9xi32>
            %67 = affine.load %20[%arg4, %arg3] : memref<9x9xi32>
            %68 = arith.muli %24, %67 : i32
            %69 = affine.load %3[%arg5, %arg3] : memref<36x9xi32>
            %70 = arith.addi %69, %68 : i32
            %71 = affine.load %21[%arg4, %arg3] : memref<9x9xi32>
            %72 = arith.muli %29, %71 : i32
            %73 = arith.addi %70, %72 : i32
            %74 = affine.load %22[%arg4, %arg3] : memref<9x9xi32>
            %75 = arith.muli %33, %74 : i32
            %76 = arith.addi %73, %75 : i32
            %77 = affine.load %23[%arg4, %arg3] : memref<9x9xi32>
            %78 = arith.muli %37, %77 : i32
            %79 = arith.addi %76, %78 : i32
            affine.store %79, %3[%arg5, %arg3] : memref<36x9xi32>
          }
        }
      }
      ADORA.terminator
    } {KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.BlockStore %3, %arg2 [0, 3] : memref<36x9xi32> -> memref<?x36xi32>  {Id = "3", KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.BlockStore %2, %arg2 [0, 1] : memref<36x9xi32> -> memref<?x36xi32>  {Id = "2", KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.BlockStore %1, %arg2 [0, 2] : memref<36x9xi32> -> memref<?x36xi32>  {Id = "1", KernelName = "merge_MATMUL_4x4_IS"}
    ADORA.BlockStore %0, %arg2 [0, 0] : memref<36x9xi32> -> memref<?x36xi32>  {Id = "0", KernelName = "merge_MATMUL_4x4_IS"}
    return
  }
}

