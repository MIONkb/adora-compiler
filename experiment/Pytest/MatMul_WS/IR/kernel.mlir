module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @merge_MATMUL_4x4_IS(%arg0: memref<?x36xi32>, %arg1: memref<?x36xi32>, %arg2: memref<?x36xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg3 = 0 to 9 {
      affine.for %arg4 = 0 to 36 {
        affine.for %arg5 = 0 to 36 {
          %0 = affine.load %arg0[%arg5, %arg4] : memref<?x36xi32>
          %1 = affine.load %arg1[%arg4, %arg3] : memref<?x36xi32>
          %2 = arith.muli %0, %1 : i32
          %3 = affine.load %arg2[%arg5, %arg3] : memref<?x36xi32>
          %4 = arith.addi %3, %2 : i32
          affine.store %4, %arg2[%arg5, %arg3] : memref<?x36xi32>
          %5 = affine.load %arg0[%arg5, %arg4 + 1] : memref<?x36xi32>
          %6 = affine.load %arg1[%arg4 + 1, %arg3 + 1] : memref<?x36xi32>
          %7 = arith.muli %5, %6 : i32
          %8 = arith.addi %4, %7 : i32
          affine.store %8, %arg2[%arg5, %arg3] : memref<?x36xi32>
          %9 = affine.load %arg0[%arg5, %arg4 + 2] : memref<?x36xi32>
          %10 = affine.load %arg1[%arg4 + 2, %arg3 + 2] : memref<?x36xi32>
          %11 = arith.muli %9, %10 : i32
          %12 = arith.addi %8, %11 : i32
          affine.store %12, %arg2[%arg5, %arg3] : memref<?x36xi32>
          %13 = affine.load %arg0[%arg5, %arg4 + 3] : memref<?x36xi32>
          %14 = affine.load %arg1[%arg4 + 3, %arg3 + 3] : memref<?x36xi32>
          %15 = arith.muli %13, %14 : i32
          %16 = arith.addi %12, %15 : i32
          affine.store %16, %arg2[%arg5, %arg3] : memref<?x36xi32>
        }
      }
    }
    return
  }
}
