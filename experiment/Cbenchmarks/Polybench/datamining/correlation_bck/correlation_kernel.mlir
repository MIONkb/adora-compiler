#map = affine_map<(d0) -> (-d0 + 1199)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("%0.2f \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("corr\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("begin dump: %s\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @kernel_correlation(%arg0: f32, %arg1: memref<?x1200xf32>, %arg2: memref<?x1200xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e-01 : f32
    affine.for %arg5 = 0 to 1200 {
      affine.store %cst_0, %arg3[%arg5] : memref<?xf32>
      affine.for %arg6 = 0 to 1400 {
        %3 = affine.load %arg1[%arg6, %arg5] : memref<?x1200xf32>
        %4 = affine.load %arg3[%arg5] : memref<?xf32>
        %5 = arith.addf %4, %3 : f32
        affine.store %5, %arg3[%arg5] : memref<?xf32>
      }
      %1 = affine.load %arg3[%arg5] : memref<?xf32>
      %2 = arith.divf %1, %arg0 : f32
      affine.store %2, %arg3[%arg5] : memref<?xf32>
    }
    affine.for %arg5 = 0 to 1200 {
      affine.store %cst_0, %arg4[%arg5] : memref<?xf32>
      affine.for %arg6 = 0 to 1400 {
        %6 = affine.load %arg1[%arg6, %arg5] : memref<?x1200xf32>
        %7 = affine.load %arg3[%arg5] : memref<?xf32>
        %8 = arith.subf %6, %7 : f32
        %9 = arith.mulf %8, %8 : f32
        %10 = affine.load %arg4[%arg5] : memref<?xf32>
        %11 = arith.addf %10, %9 : f32
        affine.store %11, %arg4[%arg5] : memref<?xf32>
      }
      %1 = affine.load %arg4[%arg5] : memref<?xf32>
      %2 = arith.divf %1, %arg0 : f32
      %3 = math.sqrt %2 : f32
      %4 = arith.cmpf ole, %3, %cst_1 : f32
      %5 = arith.select %4, %cst, %3 : f32
      affine.store %5, %arg4[%arg5] : memref<?xf32>
    }
    %0 = math.sqrt %arg0 : f32
    affine.for %arg5 = 0 to 1400 {
      affine.for %arg6 = 0 to 1200 {
        %1 = affine.load %arg3[%arg6] : memref<?xf32>
        %2 = affine.load %arg1[%arg5, %arg6] : memref<?x1200xf32>
        %3 = arith.subf %2, %1 : f32
        affine.store %3, %arg1[%arg5, %arg6] : memref<?x1200xf32>
        %4 = affine.load %arg4[%arg6] : memref<?xf32>
        %5 = arith.mulf %0, %4 : f32
        %6 = arith.divf %3, %5 : f32
        affine.store %6, %arg1[%arg5, %arg6] : memref<?x1200xf32>
      }
    }
    affine.for %arg5 = 0 to 1199 {
      affine.store %cst, %arg2[%arg5, %arg5] : memref<?x1200xf32>
      affine.for %arg6 = 0 to #map(%arg5) {
        affine.store %cst_0, %arg2[%arg5, %arg5 + %arg6 + 1] : memref<?x1200xf32>
        affine.for %arg7 = 0 to 1400 {
          %2 = affine.load %arg1[%arg7, %arg5] : memref<?x1200xf32>
          %3 = affine.load %arg1[%arg7, %arg5 + %arg6 + 1] : memref<?x1200xf32>
          %4 = arith.mulf %2, %3 : f32
          %5 = affine.load %arg2[%arg5, %arg5 + %arg6 + 1] : memref<?x1200xf32>
          %6 = arith.addf %5, %4 : f32
          affine.store %6, %arg2[%arg5, %arg5 + %arg6 + 1] : memref<?x1200xf32>
        }
        %1 = affine.load %arg2[%arg5, %arg5 + %arg6 + 1] : memref<?x1200xf32>
        affine.store %1, %arg2[%arg5 + %arg6 + 1, %arg5] : memref<?x1200xf32>
      }
    }
    affine.store %cst, %arg2[1199, 1199] : memref<?x1200xf32>
    return
  }
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1200_i32 = arith.constant 1200 : i32
    %c1400_i32 = arith.constant 1400 : i32
    %alloca = memref.alloca() : memref<1200xf32>
    %alloca_0 = memref.alloca() : memref<1200xf32>
    %alloca_1 = memref.alloca() : memref<1200x1200xf32>
    %alloca_2 = memref.alloca() : memref<1400x1200xf32>
    %alloca_3 = memref.alloca() : memref<1xf32>
    %0 = llvm.mlir.undef : f32
    affine.store %0, %alloca_3[0] : memref<1xf32>
    %cast = memref.cast %alloca_3 : memref<1xf32> to memref<?xf32>
    %cast_4 = memref.cast %alloca_2 : memref<1400x1200xf32> to memref<?x1200xf32>
    call @init_array(%c1200_i32, %c1400_i32, %cast, %cast_4) : (i32, i32, memref<?xf32>, memref<?x1200xf32>) -> ()
    %1 = affine.load %alloca_3[0] : memref<1xf32>
    %cast_5 = memref.cast %alloca_1 : memref<1200x1200xf32> to memref<?x1200xf32>
    %cast_6 = memref.cast %alloca_0 : memref<1200xf32> to memref<?xf32>
    %cast_7 = memref.cast %alloca : memref<1200xf32> to memref<?xf32>
    call @kernel_correlation(%1, %cast_4, %cast_5, %cast_6, %cast_7) : (f32, memref<?x1200xf32>, memref<?x1200xf32>, memref<?xf32>, memref<?xf32>) -> ()
    call @print_array(%c1200_i32, %cast_5) : (i32, memref<?x1200xf32>) -> ()
    return %c0_i32 : i32
  }
  func.func private @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?x1200xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c1400 = arith.constant 1400 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1200 = arith.constant 1200 : index
    %cst = arith.constant 1.200000e+03 : f32
    %cst_0 = arith.constant 1.400000e+03 : f32
    affine.store %cst_0, %arg2[0] : memref<?xf32>
    scf.for %arg4 = %c0 to %c1400 step %c1 {
      %0 = arith.index_cast %arg4 : index to i32
      scf.for %arg5 = %c0 to %c1200 step %c1 {
        %1 = arith.index_cast %arg5 : index to i32
        %2 = arith.muli %0, %1 : i32
        %3 = arith.sitofp %2 : i32 to f32
        %4 = arith.divf %3, %cst : f32
        %5 = arith.sitofp %0 : i32 to f32
        %6 = arith.addf %4, %5 : f32
        memref.store %6, %arg3[%arg4, %arg5] : memref<?x1200xf32>
      }
    }
    return
  }
  func.func private @print_array(%arg0: i32, %arg1: memref<?x1200xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c20_i32 = arith.constant 20 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.addressof @str0 : !llvm.ptr
    %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
    %2 = llvm.call @printf(%1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %3 = llvm.mlir.addressof @str1 : !llvm.ptr
    %4 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
    %5 = llvm.mlir.addressof @str2 : !llvm.ptr
    %6 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
    %7 = llvm.call @printf(%4, %6) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %8 = arith.index_cast %arg0 : i32 to index
    scf.for %arg2 = %c0 to %8 step %c1 {
      %15 = arith.index_cast %arg2 : index to i32
      scf.for %arg3 = %c0 to %8 step %c1 {
        %16 = arith.index_cast %arg3 : index to i32
        %17 = arith.muli %15, %arg0 : i32
        %18 = arith.addi %17, %16 : i32
        %19 = arith.remsi %18, %c20_i32 : i32
        %20 = arith.cmpi eq, %19, %c0_i32 : i32
        scf.if %20 {
          %26 = llvm.mlir.addressof @str3 : !llvm.ptr
          %27 = llvm.getelementptr %26[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
          %28 = llvm.call @printf(%27) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        }
        %21 = llvm.mlir.addressof @str4 : !llvm.ptr
        %22 = llvm.getelementptr %21[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x i8>
        %23 = memref.load %arg1[%arg2, %arg3] : memref<?x1200xf32>
        %24 = arith.extf %23 : f32 to f64
        %25 = llvm.call @printf(%22, %24) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
      }
    }
    %9 = llvm.mlir.addressof @str5 : !llvm.ptr
    %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    %11 = llvm.call @printf(%10, %6) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %12 = llvm.mlir.addressof @str6 : !llvm.ptr
    %13 = llvm.getelementptr %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
    %14 = llvm.call @printf(%13) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    return
  }
}

