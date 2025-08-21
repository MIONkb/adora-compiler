module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @merge_MATMUL_4x4_IS(%arg0: memref<?x36xi32>, %arg1: memref<?x36xi32>, %arg2: memref<?x36xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg3 = 0 to 9 {
      affine.for %arg4 = 0 to 9 {
        affine.for %arg5 = 0 to 36 {
          %0 = affine.load %arg0[%arg5, %arg4] : memref<?x36xi32>
          %1 = affine.load %arg1[%arg4, %arg3] : memref<?x36xi32>
          %2 = arith.muli %0, %1 : i32
          %3 = affine.load %arg2[%arg5, %arg3] : memref<?x36xi32>
          %4 = arith.addi %3, %2 : i32
          affine.store %4, %arg2[%arg5, %arg3] : memref<?x36xi32>
          %5 = affine.load %arg0[%arg5, %arg4 + 1] : memref<?x36xi32>
          %6 = affine.load %arg1[%arg4 + 1, %arg3] : memref<?x36xi32>
          %7 = arith.muli %5, %6 : i32
          %8 = arith.addi %4, %7 : i32
          affine.store %8, %arg2[%arg5, %arg3] : memref<?x36xi32>
          %9 = affine.load %arg0[%arg5, %arg4 + 2] : memref<?x36xi32>
          %10 = affine.load %arg1[%arg4 + 2, %arg3] : memref<?x36xi32>
          %11 = arith.muli %9, %10 : i32
          %12 = arith.addi %8, %11 : i32
          affine.store %12, %arg2[%arg5, %arg3] : memref<?x36xi32>
          %13 = affine.load %arg0[%arg5, %arg4 + 3] : memref<?x36xi32>
          %14 = affine.load %arg1[%arg4 + 3, %arg3] : memref<?x36xi32>
          %15 = arith.muli %13, %14 : i32
          %16 = arith.addi %12, %15 : i32
          affine.store %16, %arg2[%arg5, %arg3] : memref<?x36xi32>
          %17 = affine.load %arg0[%arg5, %arg4] : memref<?x36xi32>
          %18 = affine.load %arg1[%arg4, %arg3 + 1] : memref<?x36xi32>
          %19 = arith.muli %17, %18 : i32
          %20 = affine.load %arg2[%arg5, %arg3 + 1] : memref<?x36xi32>
          %21 = arith.addi %20, %19 : i32
          affine.store %21, %arg2[%arg5, %arg3 + 1] : memref<?x36xi32>
          %22 = affine.load %arg0[%arg5, %arg4 + 1] : memref<?x36xi32>
          %23 = affine.load %arg1[%arg4 + 1, %arg3 + 1] : memref<?x36xi32>
          %24 = arith.muli %22, %23 : i32
          %25 = arith.addi %21, %24 : i32
          affine.store %25, %arg2[%arg5, %arg3 + 1] : memref<?x36xi32>
          %26 = affine.load %arg0[%arg5, %arg4 + 2] : memref<?x36xi32>
          %27 = affine.load %arg1[%arg4 + 2, %arg3 + 1] : memref<?x36xi32>
          %28 = arith.muli %26, %27 : i32
          %29 = arith.addi %25, %28 : i32
          affine.store %29, %arg2[%arg5, %arg3 + 1] : memref<?x36xi32>
          %30 = affine.load %arg0[%arg5, %arg4 + 3] : memref<?x36xi32>
          %31 = affine.load %arg1[%arg4 + 3, %arg3 + 1] : memref<?x36xi32>
          %32 = arith.muli %30, %31 : i32
          %33 = arith.addi %29, %32 : i32
          affine.store %33, %arg2[%arg5, %arg3 + 1] : memref<?x36xi32>
          %34 = affine.load %arg0[%arg5, %arg4] : memref<?x36xi32>
          %35 = affine.load %arg1[%arg4, %arg3 + 2] : memref<?x36xi32>
          %36 = arith.muli %34, %35 : i32
          %37 = affine.load %arg2[%arg5, %arg3 + 2] : memref<?x36xi32>
          %38 = arith.addi %37, %36 : i32
          affine.store %38, %arg2[%arg5, %arg3 + 2] : memref<?x36xi32>
          %39 = affine.load %arg0[%arg5, %arg4 + 1] : memref<?x36xi32>
          %40 = affine.load %arg1[%arg4 + 1, %arg3 + 2] : memref<?x36xi32>
          %41 = arith.muli %39, %40 : i32
          %42 = arith.addi %38, %41 : i32
          affine.store %42, %arg2[%arg5, %arg3 + 2] : memref<?x36xi32>
          %43 = affine.load %arg0[%arg5, %arg4 + 2] : memref<?x36xi32>
          %44 = affine.load %arg1[%arg4 + 2, %arg3 + 2] : memref<?x36xi32>
          %45 = arith.muli %43, %44 : i32
          %46 = arith.addi %42, %45 : i32
          affine.store %46, %arg2[%arg5, %arg3 + 2] : memref<?x36xi32>
          %47 = affine.load %arg0[%arg5, %arg4 + 3] : memref<?x36xi32>
          %48 = affine.load %arg1[%arg4 + 3, %arg3 + 2] : memref<?x36xi32>
          %49 = arith.muli %47, %48 : i32
          %50 = arith.addi %46, %49 : i32
          affine.store %50, %arg2[%arg5, %arg3 + 2] : memref<?x36xi32>
          %51 = affine.load %arg0[%arg5, %arg4] : memref<?x36xi32>
          %52 = affine.load %arg1[%arg4, %arg3 + 3] : memref<?x36xi32>
          %53 = arith.muli %51, %52 : i32
          %54 = affine.load %arg2[%arg5, %arg3 + 3] : memref<?x36xi32>
          %55 = arith.addi %54, %53 : i32
          affine.store %55, %arg2[%arg5, %arg3 + 3] : memref<?x36xi32>
          %56 = affine.load %arg0[%arg5, %arg4 + 1] : memref<?x36xi32>
          %57 = affine.load %arg1[%arg4 + 1, %arg3 + 3] : memref<?x36xi32>
          %58 = arith.muli %56, %57 : i32
          %59 = arith.addi %55, %58 : i32
          affine.store %59, %arg2[%arg5, %arg3 + 3] : memref<?x36xi32>
          %60 = affine.load %arg0[%arg5, %arg4 + 2] : memref<?x36xi32>
          %61 = affine.load %arg1[%arg4 + 2, %arg3 + 3] : memref<?x36xi32>
          %62 = arith.muli %60, %61 : i32
          %63 = arith.addi %59, %62 : i32
          affine.store %63, %arg2[%arg5, %arg3 + 3] : memref<?x36xi32>
          %64 = affine.load %arg0[%arg5, %arg4 + 3] : memref<?x36xi32>
          %65 = affine.load %arg1[%arg4 + 3, %arg3 + 3] : memref<?x36xi32>
          %66 = arith.muli %64, %65 : i32
          %67 = arith.addi %63, %66 : i32
          affine.store %67, %arg2[%arg5, %arg3 + 3] : memref<?x36xi32>
        }
      }
    }
    return
  }
}

