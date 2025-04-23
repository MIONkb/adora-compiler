#map = affine_map<(d0) -> (-d0 + 27)>
#map1 = affine_map<(d0, d1) -> (d0 + d1 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @correlation(%arg0: f32, %arg1: memref<?x28xf32>, %arg2: memref<?x28xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 5.65685415 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e-01 : f32
    %cst_3 = arith.constant 3.200000e+01 : f32
    affine.for %arg5 = 0 to 28 {
      affine.store %cst_1, %arg3[%arg5] : memref<?xf32>
    }
    affine.for %arg5 = 0 to 32 {
      affine.for %arg6 = 0 to 28 {
        %0 = affine.load %arg1[%arg5, %arg6] : memref<?x28xf32>
        %1 = affine.load %arg3[%arg6] : memref<?xf32>
        %2 = arith.addf %1, %0 : f32
        affine.store %2, %arg3[%arg6] : memref<?xf32>
      }
    }
    affine.for %arg5 = 0 to 28 {
      %0 = affine.load %arg3[%arg5] : memref<?xf32>
      %1 = arith.divf %0, %cst_3 : f32
      affine.store %1, %arg3[%arg5] : memref<?xf32>
    }
    affine.for %arg5 = 0 to 28 {
      affine.store %cst_1, %arg4[%arg5] : memref<?xf32>
    }
    affine.for %arg5 = 0 to 32 {
      affine.for %arg6 = 0 to 28 {
        %0 = affine.load %arg1[%arg5, %arg6] : memref<?x28xf32>
        %1 = affine.load %arg3[%arg6] : memref<?xf32>
        %2 = arith.subf %0, %1 : f32
        %3 = arith.mulf %2, %2 : f32
        %4 = affine.load %arg4[%arg6] : memref<?xf32>
        %5 = arith.addf %4, %3 : f32
        affine.store %5, %arg4[%arg6] : memref<?xf32>
      }
    }
    affine.for %arg5 = 0 to 28 {
      affine.store %cst_1, %arg4[%arg5] : memref<?xf32>
      affine.for %arg6 = 0 to 32 {
        %5 = affine.load %arg1[%arg6, %arg5] : memref<?x28xf32>
        %6 = affine.load %arg3[%arg5] : memref<?xf32>
        %7 = arith.subf %5, %6 : f32
        %8 = arith.mulf %7, %7 : f32
        %9 = affine.load %arg4[%arg5] : memref<?xf32>
        %10 = arith.addf %9, %8 : f32
        affine.store %10, %arg4[%arg5] : memref<?xf32>
      }
      %0 = affine.load %arg4[%arg5] : memref<?xf32>
      %1 = arith.divf %0, %cst_3 : f32
      %2 = math.sqrt %1 : f32
      %3 = arith.cmpf ole, %2, %cst_2 : f32
      %4 = arith.select %3, %cst_0, %2 : f32
      affine.store %4, %arg4[%arg5] : memref<?xf32>
    }
    affine.for %arg5 = 0 to 32 {
      affine.for %arg6 = 0 to 28 {
        %0 = affine.load %arg3[%arg6] : memref<?xf32>
        %1 = affine.load %arg1[%arg5, %arg6] : memref<?x28xf32>
        %2 = arith.subf %1, %0 : f32
        affine.store %2, %arg1[%arg5, %arg6] : memref<?x28xf32>
        %3 = affine.load %arg4[%arg6] : memref<?xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.divf %2, %4 : f32
        affine.store %5, %arg1[%arg5, %arg6] : memref<?x28xf32>
      }
    }
    affine.for %arg5 = 0 to 27 {
      affine.store %cst_0, %arg2[%arg5, %arg5] : memref<?x28xf32>
      affine.for %arg6 = 0 to #map(%arg5) {
        %0 = affine.apply #map1(%arg5, %arg6)
        affine.store %cst_1, %arg2[%arg5, %0] : memref<?x28xf32>
        affine.for %arg7 = 0 to 32 {
          %2 = affine.load %arg1[%arg7, %arg5] : memref<?x28xf32>
          %3 = affine.load %arg1[%arg7, %0] : memref<?x28xf32>
          %4 = arith.mulf %2, %3 : f32
          %5 = affine.load %arg2[%arg5, %0] : memref<?x28xf32>
          %6 = arith.addf %5, %4 : f32
          affine.store %6, %arg2[%arg5, %0] : memref<?x28xf32>
        }
        %1 = affine.load %arg2[%arg5, %0] : memref<?x28xf32>
        affine.store %1, %arg2[%0, %arg5] : memref<?x28xf32>
      }
    }
    affine.store %cst_0, %arg2[27, 27] : memref<?x28xf32>
    return
  }
}

