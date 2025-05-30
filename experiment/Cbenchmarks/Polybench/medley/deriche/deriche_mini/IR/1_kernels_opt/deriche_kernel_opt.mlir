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
    affine.for %arg4 = 0 to 64 step 32 {
      %1 = ADORA.BlockLoad %arg0 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_0"}
      %2 = ADORA.LocalMemAlloc memref<32x64xf32>  {Id = "1", KernelName = "kernel_deriche_0"}
      %3 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "2", KernelName = "kernel_deriche_0"}
      %4 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "3", KernelName = "kernel_deriche_0"}
      %5 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "4", KernelName = "kernel_deriche_0"}
      ADORA.kernel {
        affine.for %arg5 = 0 to 32 {
          %6:3 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5, %arg8 = %cst_5, %arg9 = %cst_5) -> (f32, f32, f32) {
            %7 = affine.load %1[%arg5, %arg6] : memref<32x64xf32>
            %8 = arith.mulf %7, %cst_2 : f32
            %9 = arith.mulf %arg7, %cst : f32
            %10 = arith.addf %8, %9 : f32
            %11 = arith.mulf %arg8, %cst_3 : f32
            %12 = arith.addf %10, %11 : f32
            %13 = arith.mulf %arg9, %cst_4 : f32
            %14 = arith.addf %12, %13 : f32
            affine.store %14, %2[%arg5, %arg6] : memref<32x64xf32>
            affine.yield %7, %14, %arg8 : f32, f32, f32
          }
          affine.store %6#2, %4[0] : memref<2xf32>
          affine.store %6#1, %5[0] : memref<2xf32>
          affine.store %6#0, %3[0] : memref<2xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_deriche_0"}
      ADORA.BlockStore %5, %alloca_12 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_0"}
      ADORA.BlockStore %4, %alloca_11 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_0"}
      ADORA.BlockStore %3, %alloca_14 [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_0"}
      ADORA.BlockStore %2, %arg2 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_0"}
    }
    affine.for %arg4 = 0 to 64 step 32 {
      %1 = ADORA.BlockLoad %arg0 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_1"}
      %2 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "1", KernelName = "kernel_deriche_1"}
      %3 = ADORA.LocalMemAlloc memref<32x64xf32>  {Id = "2", KernelName = "kernel_deriche_1"}
      %4 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "3", KernelName = "kernel_deriche_1"}
      %5 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "4", KernelName = "kernel_deriche_1"}
      %6 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "5", KernelName = "kernel_deriche_1"}
      ADORA.kernel {
        affine.for %arg5 = 0 to 32 {
          %7:4 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5, %arg8 = %cst_5, %arg9 = %cst_5, %arg10 = %cst_5) -> (f32, f32, f32, f32) {
            %8 = arith.mulf %arg7, %cst_0 : f32
            %9 = arith.mulf %arg8, %cst_1 : f32
            %10 = arith.addf %8, %9 : f32
            %11 = arith.mulf %arg9, %cst_3 : f32
            %12 = arith.addf %10, %11 : f32
            %13 = arith.mulf %arg10, %cst_4 : f32
            %14 = arith.addf %12, %13 : f32
            affine.store %14, %3[%arg5, -%arg6 + 63] : memref<32x64xf32>
            %15 = affine.load %1[%arg5, -%arg6 + 63] : memref<32x64xf32>
            affine.yield %15, %arg7, %14, %arg9 : f32, f32, f32, f32
          }
          affine.store %7#3, %5[0] : memref<2xf32>
          affine.store %7#2, %6[0] : memref<2xf32>
          affine.store %7#1, %4[0] : memref<2xf32>
          affine.store %7#0, %2[0] : memref<2xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_deriche_1"}
      ADORA.BlockStore %6, %alloca_6 [] : memref<2xf32> -> memref<f32>  {Id = "5", KernelName = "kernel_deriche_1"}
      ADORA.BlockStore %5, %alloca [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_1"}
      ADORA.BlockStore %4, %alloca_9 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_1"}
      ADORA.BlockStore %3, %arg3 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "2", KernelName = "kernel_deriche_1"}
      ADORA.BlockStore %2, %alloca_10 [] : memref<2xf32> -> memref<f32>  {Id = "1", KernelName = "kernel_deriche_1"}
    }
    affine.for %arg4 = 0 to 64 step 32 {
      %1 = ADORA.BlockLoad %arg2 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_2"}
      %2 = ADORA.BlockLoad %arg3 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "1", KernelName = "kernel_deriche_2"}
      %3 = ADORA.LocalMemAlloc memref<32x64xf32>  {Id = "2", KernelName = "kernel_deriche_2"}
      ADORA.kernel {
        affine.for %arg5 = 0 to 32 {
          affine.for %arg6 = 0 to 64 {
            %4 = affine.load %1[%arg5, %arg6] : memref<32x64xf32>
            %5 = affine.load %2[%arg5, %arg6] : memref<32x64xf32>
            %6 = arith.addf %4, %5 : f32
            affine.store %6, %3[%arg5, %arg6] : memref<32x64xf32>
          }
        }
        ADORA.terminator
      } {KernelName = "kernel_deriche_2"}
      ADORA.BlockStore %3, %arg1 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "2", KernelName = "kernel_deriche_2"}
    }
    affine.for %arg4 = 0 to 64 step 32 {
      %1 = ADORA.BlockLoad %arg1 [0, %arg4] : memref<?x64xf32> -> memref<64x32xf32>  {Id = "0", KernelName = "kernel_deriche_3"}
      %2 = ADORA.LocalMemAlloc memref<64x32xf32>  {Id = "1", KernelName = "kernel_deriche_3"}
      %3 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "2", KernelName = "kernel_deriche_3"}
      %4 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "3", KernelName = "kernel_deriche_3"}
      %5 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "4", KernelName = "kernel_deriche_3"}
      ADORA.kernel {
        affine.for %arg5 = 0 to 32 {
          %6:3 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5, %arg8 = %cst_5, %arg9 = %cst_5) -> (f32, f32, f32) {
            %7 = affine.load %1[%arg6, %arg5] : memref<64x32xf32>
            %8 = arith.mulf %7, %cst_2 : f32
            %9 = arith.mulf %arg7, %cst : f32
            %10 = arith.addf %8, %9 : f32
            %11 = arith.mulf %arg8, %cst_3 : f32
            %12 = arith.addf %10, %11 : f32
            %13 = arith.mulf %arg9, %cst_4 : f32
            %14 = arith.addf %12, %13 : f32
            affine.store %14, %2[%arg6, %arg5] : memref<64x32xf32>
            affine.yield %7, %14, %arg8 : f32, f32, f32
          }
          affine.store %6#2, %3[0] : memref<2xf32>
          affine.store %6#1, %4[0] : memref<2xf32>
          affine.store %6#0, %5[0] : memref<2xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_deriche_3"}
      ADORA.BlockStore %5, %alloca_13 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_3"}
      ADORA.BlockStore %4, %alloca_12 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_3"}
      ADORA.BlockStore %3, %alloca_11 [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_3"}
      ADORA.BlockStore %2, %arg2 [0, %arg4] : memref<64x32xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_3"}
    }
    affine.for %arg4 = 0 to 64 step 32 {
      %1 = ADORA.BlockLoad %arg1 [0, %arg4] : memref<?x64xf32> -> memref<64x32xf32>  {Id = "0", KernelName = "kernel_deriche_4"}
      %2 = ADORA.LocalMemAlloc memref<64x32xf32>  {Id = "1", KernelName = "kernel_deriche_4"}
      %3 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "2", KernelName = "kernel_deriche_4"}
      %4 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "3", KernelName = "kernel_deriche_4"}
      %5 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "4", KernelName = "kernel_deriche_4"}
      %6 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "5", KernelName = "kernel_deriche_4"}
      ADORA.kernel {
        affine.for %arg5 = 0 to 32 {
          %7:4 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5, %arg8 = %cst_5, %arg9 = %cst_5, %arg10 = %cst_5) -> (f32, f32, f32, f32) {
            %8 = arith.mulf %arg7, %cst_0 : f32
            %9 = arith.mulf %arg8, %cst_1 : f32
            %10 = arith.addf %8, %9 : f32
            %11 = arith.mulf %arg9, %cst_3 : f32
            %12 = arith.addf %10, %11 : f32
            %13 = arith.mulf %arg10, %cst_4 : f32
            %14 = arith.addf %12, %13 : f32
            affine.store %14, %2[-%arg6 + 63, %arg5] : memref<64x32xf32>
            %15 = affine.load %1[-%arg6 + 63, %arg5] : memref<64x32xf32>
            affine.yield %15, %arg7, %14, %arg9 : f32, f32, f32, f32
          }
          affine.store %7#3, %3[0] : memref<2xf32>
          affine.store %7#2, %4[0] : memref<2xf32>
          affine.store %7#1, %5[0] : memref<2xf32>
          affine.store %7#0, %6[0] : memref<2xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_deriche_4"}
      ADORA.BlockStore %6, %alloca_8 [] : memref<2xf32> -> memref<f32>  {Id = "5", KernelName = "kernel_deriche_4"}
      ADORA.BlockStore %5, %alloca_7 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = "kernel_deriche_4"}
      ADORA.BlockStore %4, %alloca_6 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = "kernel_deriche_4"}
      ADORA.BlockStore %3, %alloca [] : memref<2xf32> -> memref<f32>  {Id = "2", KernelName = "kernel_deriche_4"}
      ADORA.BlockStore %2, %arg3 [0, %arg4] : memref<64x32xf32> -> memref<?x64xf32>  {Id = "1", KernelName = "kernel_deriche_4"}
    }
    affine.for %arg4 = 0 to 64 step 32 {
      %1 = ADORA.BlockLoad %arg2 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "0", KernelName = "kernel_deriche_5"}
      %2 = ADORA.BlockLoad %arg3 [%arg4, 0] : memref<?x64xf32> -> memref<32x64xf32>  {Id = "1", KernelName = "kernel_deriche_5"}
      %3 = ADORA.LocalMemAlloc memref<32x64xf32>  {Id = "2", KernelName = "kernel_deriche_5"}
      ADORA.kernel {
        affine.for %arg5 = 0 to 32 {
          affine.for %arg6 = 0 to 64 {
            %4 = affine.load %1[%arg5, %arg6] : memref<32x64xf32>
            %5 = affine.load %2[%arg5, %arg6] : memref<32x64xf32>
            %6 = arith.addf %4, %5 : f32
            affine.store %6, %3[%arg5, %arg6] : memref<32x64xf32>
          }
        }
        ADORA.terminator
      } {KernelName = "kernel_deriche_5"}
      ADORA.BlockStore %3, %arg1 [%arg4, 0] : memref<32x64xf32> -> memref<?x64xf32>  {Id = "2", KernelName = "kernel_deriche_5"}
    }
    return
  }
}

