#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("%0.2lf \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("corr\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("begin dump: %s\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 37.416573867739416 : f64
    %c0 = arith.constant 0 : index
    %c1200 = arith.constant 1200 : index
    %c1400 = arith.constant 1400 : index
    %c20_i32 = arith.constant 20 : i32
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %cst_2 = arith.constant 1.000000e-01 : f64
    %cst_3 = arith.constant 1.400000e+03 : f64
    %cst_4 = arith.constant 1.200000e+03 : f64
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c1200_i32 = arith.constant 1200 : i32
    %alloc = memref.alloc() : memref<1400x1200xf64>
    %alloc_5 = memref.alloc() : memref<1200x1200xf64>
    %alloc_6 = memref.alloc() : memref<1200xf64>
    %alloc_7 = memref.alloc() : memref<1200xf64>
    scf.for %arg2 = %c0 to %c1400 step %c1 {
      %18 = arith.index_cast %arg2 : index to i32
      %19 = arith.sitofp %18 : i32 to f64
      scf.for %arg3 = %c0 to %c1200 step %c1 {
        %20 = arith.index_cast %arg3 : index to i32
        %21 = arith.muli %18, %20 : i32
        %22 = arith.sitofp %21 : i32 to f64
        %23 = arith.divf %22, %cst_4 : f64
        %24 = arith.addf %23, %19 : f64
        memref.store %24, %alloc[%arg2, %arg3] : memref<1400x1200xf64>
      }
    }


    /// initialization
    affine.for %arg2 = 0 to 1200 {
      affine.store %cst_1, %alloc_6[%arg2] : memref<1200xf64>
      %18 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 1400 {
        %21 = arith.index_cast %arg3 : index to i32
        %22 = arith.muli %21, %18 : i32
        %23 = arith.sitofp %22 : i32 to f64
        %24 = arith.divf %23, %cst_4 : f64
        %25 = arith.sitofp %21 : i32 to f64
        %26 = arith.addf %24, %25 : f64
        %27 = affine.load %alloc_6[%arg2] : memref<1200xf64>
        %28 = arith.addf %27, %26 : f64
        affine.store %28, %alloc_6[%arg2] : memref<1200xf64>
      }
      %19 = affine.load %alloc_6[%arg2] : memref<1200xf64>
      %20 = arith.divf %19, %cst_3 : f64
      affine.store %20, %alloc_6[%arg2] : memref<1200xf64>
    }


    /// Start computing
    affine.for %arg2 = 0 to 1200 {
      affine.store %cst_1, %alloc_7[%arg2] : memref<1200xf64>
      %18 = arith.index_cast %arg2 : index to i32
      %19 = affine.load %alloc_6[%arg2] : memref<1200xf64>
      affine.for %arg3 = 0 to 1400 {
        %25 = arith.index_cast %arg3 : index to i32
        %26 = arith.muli %25, %18 : i32
        %27 = arith.sitofp %26 : i32 to f64
        %28 = arith.divf %27, %cst_4 : f64
        %29 = arith.sitofp %25 : i32 to f64
        %30 = arith.addf %28, %29 : f64
        %31 = arith.subf %30, %19 : f64
        %32 = arith.mulf %31, %31 : f64
        %33 = affine.load %alloc_7[%arg2] : memref<1200xf64>
        %34 = arith.addf %33, %32 : f64
        affine.store %34, %alloc_7[%arg2] : memref<1200xf64>
      }
      %20 = affine.load %alloc_7[%arg2] : memref<1200xf64>
      %21 = arith.divf %20, %cst_3 : f64
      %22 = math.sqrt %21 : f64
      %23 = arith.cmpf ole, %22, %cst_2 : f64
      %24 = arith.select %23, %cst_0, %22 : f64
      affine.store %24, %alloc_7[%arg2] : memref<1200xf64>
    }
    affine.for %arg2 = 0 to 1400 {
      affine.for %arg3 = 0 to 1200 {
        %18 = affine.load %alloc_6[%arg3] : memref<1200xf64>
        %19 = affine.load %alloc[%arg2, %arg3] : memref<1400x1200xf64>
        %20 = arith.subf %19, %18 : f64
        affine.store %20, %alloc[%arg2, %arg3] : memref<1400x1200xf64>
        %21 = affine.load %alloc_7[%arg3] : memref<1200xf64>
        %22 = arith.mulf %21, %cst : f64
        %23 = arith.divf %20, %22 : f64
        affine.store %23, %alloc[%arg2, %arg3] : memref<1400x1200xf64>
      }
    }
    affine.for %arg2 = 0 to 1199 {
      affine.store %cst_0, %alloc_5[%arg2, %arg2] : memref<1200x1200xf64>
      affine.for %arg3 = #map(%arg2) to 1200 {
        affine.store %cst_1, %alloc_5[%arg2, %arg3] : memref<1200x1200xf64>
        affine.for %arg4 = 0 to 1400 {
          %19 = affine.load %alloc[%arg4, %arg2] : memref<1400x1200xf64>
          %20 = affine.load %alloc[%arg4, %arg3] : memref<1400x1200xf64>
          %21 = arith.mulf %19, %20 : f64
          %22 = affine.load %alloc_5[%arg2, %arg3] : memref<1200x1200xf64>
          %23 = arith.addf %22, %21 : f64
          affine.store %23, %alloc_5[%arg2, %arg3] : memref<1200x1200xf64>
        }
        %18 = affine.load %alloc_5[%arg2, %arg3] : memref<1200x1200xf64>
        affine.store %18, %alloc_5[%arg3, %arg2] : memref<1200x1200xf64>
      }
    }
    affine.store %cst_0, %alloc_5[1199, 1199] : memref<1200x1200xf64>
    %0 = llvm.mlir.addressof @str0 : !llvm.ptr
    %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
    %2 = llvm.call @printf(%1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %3 = llvm.mlir.addressof @str1 : !llvm.ptr
    %4 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
    %5 = llvm.mlir.addressof @str2 : !llvm.ptr
    %6 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
    %7 = llvm.call @printf(%4, %6) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %8 = llvm.mlir.addressof @str4 : !llvm.ptr
    %9 = llvm.getelementptr %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
    %10 = llvm.mlir.addressof @str3 : !llvm.ptr
    %11 = llvm.getelementptr %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    scf.for %arg2 = %c0 to %c1200 step %c1 {
      %18 = arith.index_cast %arg2 : index to i32
      %19 = arith.muli %18, %c1200_i32 : i32
      scf.for %arg3 = %c0 to %c1200 step %c1 {
        %20 = arith.index_cast %arg3 : index to i32
        %21 = arith.addi %19, %20 : i32
        %22 = arith.remsi %21, %c20_i32 : i32
        %23 = arith.cmpi eq, %22, %c0_i32 : i32
        scf.if %23 {
          %26 = llvm.call @printf(%11) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        }
        %24 = memref.load %alloc_5[%arg2, %arg3] : memref<1200x1200xf64>
        %25 = llvm.call @printf(%9, %24) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
      }
    }
    %12 = llvm.mlir.addressof @str5 : !llvm.ptr
    %13 = llvm.getelementptr %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    %14 = llvm.call @printf(%13, %6) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %15 = llvm.mlir.addressof @str6 : !llvm.ptr
    %16 = llvm.getelementptr %15[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
    %17 = llvm.call @printf(%16) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    memref.dealloc %alloc : memref<1400x1200xf64>
    memref.dealloc %alloc_5 : memref<1200x1200xf64>
    memref.dealloc %alloc_6 : memref<1200xf64>
    memref.dealloc %alloc_7 : memref<1200xf64>
    return %c0_i32 : i32
  }
}
