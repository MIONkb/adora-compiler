module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: memref<?x18xf32>, %arg1: memref<?x20xf32>, %arg2: memref<?x18xf32>, %arg3: memref<?x22xf32>, %arg4: memref<?x24xf32>, %arg5: memref<?x22xf32>, %arg6: memref<?x22xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg7 = 0 to 16 {
      affine.for %arg8 = 0 to 18 {
        affine.store %cst, %arg0[%arg7, %arg8] : memref<?x18xf32>
        affine.for %arg9 = 0 to 20 {
          %0 = affine.load %arg1[%arg7, %arg9] : memref<?x20xf32>
          %1 = affine.load %arg2[%arg9, %arg8] : memref<?x18xf32>
          %2 = arith.mulf %0, %1 : f32
          %3 = affine.load %arg0[%arg7, %arg8] : memref<?x18xf32>
          %4 = arith.addf %3, %2 : f32
          affine.store %4, %arg0[%arg7, %arg8] : memref<?x18xf32>
        }
      }
    }
    affine.for %arg7 = 0 to 18 {
      affine.for %arg8 = 0 to 22 {
        affine.store %cst, %arg3[%arg7, %arg8] : memref<?x22xf32>
        affine.for %arg9 = 0 to 24 {
          %0 = affine.load %arg4[%arg7, %arg9] : memref<?x24xf32>
          %1 = affine.load %arg5[%arg9, %arg8] : memref<?x22xf32>
          %2 = arith.mulf %0, %1 : f32
          %3 = affine.load %arg3[%arg7, %arg8] : memref<?x22xf32>
          %4 = arith.addf %3, %2 : f32
          affine.store %4, %arg3[%arg7, %arg8] : memref<?x22xf32>
        }
      }
    }
    affine.for %arg7 = 0 to 16 {
      affine.for %arg8 = 0 to 22 {
        affine.store %cst, %arg6[%arg7, %arg8] : memref<?x22xf32>
        affine.for %arg9 = 0 to 18 {
          %0 = affine.load %arg0[%arg7, %arg9] : memref<?x18xf32>
          %1 = affine.load %arg3[%arg9, %arg8] : memref<?x22xf32>
          %2 = arith.mulf %0, %1 : f32
          %3 = affine.load %arg6[%arg7, %arg8] : memref<?x22xf32>
          %4 = arith.addf %3, %2 : f32
          affine.store %4, %arg6[%arg7, %arg8] : memref<?x22xf32>
        }
      }
    }
    return
  }
}

