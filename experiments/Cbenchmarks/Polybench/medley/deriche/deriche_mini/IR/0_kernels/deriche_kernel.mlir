module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_deriche(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>, %arg2: memref<?x64xf32>, %arg3: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.110209078 : f32
    %cst_0 = arith.constant -0.183681786 : f32
    %cst_1 = arith.constant 0.114441216 : f32
    %cst_2 = arith.constant -0.188681662 : f32
    %cst_3 = arith.constant 0.840896427 : f32
    %cst_4 = arith.constant -0.606530666 : f32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %0, %alloca[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_8[] : memref<f32>
    %alloca_9 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_9[] : memref<f32>
    %alloca_10 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_10[] : memref<f32>
    %alloca_11 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_11[] : memref<f32>
    %alloca_12 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_12[] : memref<f32>
    %alloca_13 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_13[] : memref<f32>
    %alloca_14 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_14[] : memref<f32>
    ADORA.kernel {
      affine.for %arg4 = 0 to 64 {
        affine.store %cst_5, %alloca_12[] : memref<f32>
        affine.store %cst_5, %alloca_11[] : memref<f32>
        affine.store %cst_5, %alloca_14[] : memref<f32>
        affine.for %arg5 = 0 to 64 {
          %1 = affine.load %arg0[%arg4, %arg5] : memref<?x64xf32>
          %2 = arith.mulf %1, %cst_2 : f32
          %3 = affine.load %alloca_14[] : memref<f32>
          %4 = arith.mulf %3, %cst : f32
          %5 = arith.addf %2, %4 : f32
          %6 = affine.load %alloca_12[] : memref<f32>
          %7 = arith.mulf %6, %cst_3 : f32
          %8 = arith.addf %5, %7 : f32
          %9 = affine.load %alloca_11[] : memref<f32>
          %10 = arith.mulf %9, %cst_4 : f32
          %11 = arith.addf %8, %10 : f32
          affine.store %11, %arg2[%arg4, %arg5] : memref<?x64xf32>
          %12 = affine.load %arg0[%arg4, %arg5] : memref<?x64xf32>
          affine.store %12, %alloca_14[] : memref<f32>
          affine.store %6, %alloca_11[] : memref<f32>
          %13 = affine.load %arg2[%arg4, %arg5] : memref<?x64xf32>
          affine.store %13, %alloca_12[] : memref<f32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_deriche_0"}
    ADORA.kernel {
      affine.for %arg4 = 0 to 64 {
        affine.store %cst_5, %alloca_6[] : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.store %cst_5, %alloca_10[] : memref<f32>
        affine.store %cst_5, %alloca_9[] : memref<f32>
        affine.for %arg5 = 0 to 64 {
          %1 = affine.load %alloca_10[] : memref<f32>
          %2 = arith.mulf %1, %cst_0 : f32
          %3 = affine.load %alloca_9[] : memref<f32>
          %4 = arith.mulf %3, %cst_1 : f32
          %5 = arith.addf %2, %4 : f32
          %6 = affine.load %alloca_6[] : memref<f32>
          %7 = arith.mulf %6, %cst_3 : f32
          %8 = arith.addf %5, %7 : f32
          %9 = affine.load %alloca[] : memref<f32>
          %10 = arith.mulf %9, %cst_4 : f32
          %11 = arith.addf %8, %10 : f32
          affine.store %11, %arg3[%arg4, -%arg5 + 63] : memref<?x64xf32>
          affine.store %1, %alloca_9[] : memref<f32>
          %12 = affine.load %arg0[%arg4, -%arg5 + 63] : memref<?x64xf32>
          affine.store %12, %alloca_10[] : memref<f32>
          affine.store %6, %alloca[] : memref<f32>
          %13 = affine.load %arg3[%arg4, -%arg5 + 63] : memref<?x64xf32>
          affine.store %13, %alloca_6[] : memref<f32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_deriche_1"}
    ADORA.kernel {
      affine.for %arg4 = 0 to 64 {
        affine.for %arg5 = 0 to 64 {
          %1 = affine.load %arg2[%arg4, %arg5] : memref<?x64xf32>
          %2 = affine.load %arg3[%arg4, %arg5] : memref<?x64xf32>
          %3 = arith.addf %1, %2 : f32
          affine.store %3, %arg1[%arg4, %arg5] : memref<?x64xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_deriche_2"}
    ADORA.kernel {
      affine.for %arg4 = 0 to 64 {
        affine.store %cst_5, %alloca_13[] : memref<f32>
        affine.store %cst_5, %alloca_12[] : memref<f32>
        affine.store %cst_5, %alloca_11[] : memref<f32>
        affine.for %arg5 = 0 to 64 {
          %1 = affine.load %arg1[%arg5, %arg4] : memref<?x64xf32>
          %2 = arith.mulf %1, %cst_2 : f32
          %3 = affine.load %alloca_13[] : memref<f32>
          %4 = arith.mulf %3, %cst : f32
          %5 = arith.addf %2, %4 : f32
          %6 = affine.load %alloca_12[] : memref<f32>
          %7 = arith.mulf %6, %cst_3 : f32
          %8 = arith.addf %5, %7 : f32
          %9 = affine.load %alloca_11[] : memref<f32>
          %10 = arith.mulf %9, %cst_4 : f32
          %11 = arith.addf %8, %10 : f32
          affine.store %11, %arg2[%arg5, %arg4] : memref<?x64xf32>
          %12 = affine.load %arg1[%arg5, %arg4] : memref<?x64xf32>
          affine.store %12, %alloca_13[] : memref<f32>
          affine.store %6, %alloca_11[] : memref<f32>
          %13 = affine.load %arg2[%arg5, %arg4] : memref<?x64xf32>
          affine.store %13, %alloca_12[] : memref<f32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_deriche_3"}
    ADORA.kernel {
      affine.for %arg4 = 0 to 64 {
        affine.store %cst_5, %alloca_8[] : memref<f32>
        affine.store %cst_5, %alloca_7[] : memref<f32>
        affine.store %cst_5, %alloca_6[] : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg5 = 0 to 64 {
          %1 = affine.load %alloca_8[] : memref<f32>
          %2 = arith.mulf %1, %cst_0 : f32
          %3 = affine.load %alloca_7[] : memref<f32>
          %4 = arith.mulf %3, %cst_1 : f32
          %5 = arith.addf %2, %4 : f32
          %6 = affine.load %alloca_6[] : memref<f32>
          %7 = arith.mulf %6, %cst_3 : f32
          %8 = arith.addf %5, %7 : f32
          %9 = affine.load %alloca[] : memref<f32>
          %10 = arith.mulf %9, %cst_4 : f32
          %11 = arith.addf %8, %10 : f32
          affine.store %11, %arg3[-%arg5 + 63, %arg4] : memref<?x64xf32>
          affine.store %1, %alloca_7[] : memref<f32>
          %12 = affine.load %arg1[-%arg5 + 63, %arg4] : memref<?x64xf32>
          affine.store %12, %alloca_8[] : memref<f32>
          affine.store %6, %alloca[] : memref<f32>
          %13 = affine.load %arg3[-%arg5 + 63, %arg4] : memref<?x64xf32>
          affine.store %13, %alloca_6[] : memref<f32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_deriche_4"}
    ADORA.kernel {
      affine.for %arg4 = 0 to 64 {
        affine.for %arg5 = 0 to 64 {
          %1 = affine.load %arg2[%arg4, %arg5] : memref<?x64xf32>
          %2 = affine.load %arg3[%arg4, %arg5] : memref<?x64xf32>
          %3 = arith.addf %1, %2 : f32
          affine.store %3, %arg1[%arg4, %arg5] : memref<?x64xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_deriche_5"}
    return
  }
}

