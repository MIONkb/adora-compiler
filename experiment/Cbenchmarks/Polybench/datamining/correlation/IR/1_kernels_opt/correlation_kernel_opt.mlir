#map = affine_map<(d0) -> (-d0 + 27)>
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
    %0 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x28xf32> -> memref<32x28xf32>  {Id = "0", KernelName = "correlation_0"}
    %1 = ADORA.BlockLoad %arg3 [0] : memref<?xf32> -> memref<28xf32>  {Id = "1", KernelName = "correlation_0"}
    %2 = ADORA.LocalMemAlloc memref<28xf32>  {Id = "2", KernelName = "correlation_0"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 28 {
          %19 = affine.load %0[%arg5, %arg6] : memref<32x28xf32>
          %20 = affine.load %1[%arg6] : memref<28xf32>
          %21 = arith.addf %20, %19 : f32
          affine.store %21, %2[%arg6] : memref<28xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "correlation_0"}
    ADORA.BlockStore %2, %arg3 [0] : memref<28xf32> -> memref<?xf32>  {Id = "2", KernelName = "correlation_0"}
    %3 = ADORA.BlockLoad %arg3 [0] : memref<?xf32> -> memref<28xf32>  {Id = "0", KernelName = "correlation_1"}
    %4 = ADORA.LocalMemAlloc memref<28xf32>  {Id = "1", KernelName = "correlation_1"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 28 {
        %19 = affine.load %3[%arg5] : memref<28xf32>
        %20 = arith.divf %19, %cst_3 : f32
        affine.store %20, %4[%arg5] : memref<28xf32>
      }
      ADORA.terminator
    } {KernelName = "correlation_1"}
    ADORA.BlockStore %4, %arg3 [0] : memref<28xf32> -> memref<?xf32>  {Id = "1", KernelName = "correlation_1"}
    affine.for %arg5 = 0 to 28 {
      affine.store %cst_1, %arg4[%arg5] : memref<?xf32>
    }
    %5 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x28xf32> -> memref<32x28xf32>  {Id = "0", KernelName = "correlation_2"}
    %6 = ADORA.BlockLoad %arg3 [0] : memref<?xf32> -> memref<28xf32>  {Id = "1", KernelName = "correlation_2"}
    %7 = ADORA.BlockLoad %arg4 [0] : memref<?xf32> -> memref<28xf32>  {Id = "2", KernelName = "correlation_2"}
    %8 = ADORA.LocalMemAlloc memref<28xf32>  {Id = "3", KernelName = "correlation_2"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 28 {
          %19 = affine.load %5[%arg5, %arg6] : memref<32x28xf32>
          %20 = affine.load %6[%arg6] : memref<28xf32>
          %21 = arith.subf %19, %20 : f32
          %22 = arith.mulf %21, %21 : f32
          %23 = affine.load %7[%arg6] : memref<28xf32>
          %24 = arith.addf %23, %22 : f32
          affine.store %24, %8[%arg6] : memref<28xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "correlation_2"}
    ADORA.BlockStore %8, %arg4 [0] : memref<28xf32> -> memref<?xf32>  {Id = "3", KernelName = "correlation_2"}
    %9 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x28xf32> -> memref<32x28xf32>  {Id = "0", KernelName = "correlation_3"}
    %10 = ADORA.BlockLoad %arg3 [0] : memref<?xf32> -> memref<28xf32>  {Id = "1", KernelName = "correlation_3"}
    %11 = ADORA.LocalMemAlloc memref<28xf32>  {Id = "2", KernelName = "correlation_3"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 28 {
        %19 = affine.for %arg6 = 0 to 32 iter_args(%arg7 = %cst_1) -> (f32) {
          %24 = affine.load %9[%arg6, %arg5] : memref<32x28xf32>
          %25 = affine.load %10[%arg5] : memref<28xf32>
          %26 = arith.subf %24, %25 : f32
          %27 = arith.mulf %26, %26 : f32
          %28 = arith.addf %arg7, %27 : f32
          affine.yield %28 : f32
        }
        %20 = arith.divf %19, %cst_3 : f32
        %21 = math.sqrt %20 : f32
        %22 = arith.cmpf ole, %21, %cst_2 : f32
        %23 = arith.select %22, %cst_0, %21 : f32
        affine.store %23, %11[%arg5] : memref<28xf32>
      }
      ADORA.terminator
    } {KernelName = "correlation_3"}
    ADORA.BlockStore %11, %arg4 [0] : memref<28xf32> -> memref<?xf32>  {Id = "2", KernelName = "correlation_3"}
    %12 = ADORA.BlockLoad %arg3 [0] : memref<?xf32> -> memref<28xf32>  {Id = "0", KernelName = "correlation_4"}
    %13 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x28xf32> -> memref<32x28xf32>  {Id = "1", KernelName = "correlation_4"}
    %14 = ADORA.BlockLoad %arg4 [0] : memref<?xf32> -> memref<28xf32>  {Id = "2", KernelName = "correlation_4"}
    %15 = ADORA.LocalMemAlloc memref<32x28xf32>  {Id = "3", KernelName = "correlation_4"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 28 {
          %19 = affine.load %12[%arg6] : memref<28xf32>
          %20 = affine.load %13[%arg5, %arg6] : memref<32x28xf32>
          %21 = arith.subf %20, %19 : f32
          %22 = affine.load %14[%arg6] : memref<28xf32>
          %23 = arith.mulf %22, %cst : f32
          %24 = arith.divf %21, %23 : f32
          affine.store %24, %15[%arg5, %arg6] : memref<32x28xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "correlation_4"}
    ADORA.BlockStore %15, %arg1 [0, 0] : memref<32x28xf32> -> memref<?x28xf32>  {Id = "3", KernelName = "correlation_4"}
    %16 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x28xf32> -> memref<32x27xf32>  {Id = "0", KernelName = "correlation_5"}
    %17 = ADORA.BlockLoad %arg1 [0, 1] : memref<?x28xf32> -> memref<32x27xf32>  {Id = "1", KernelName = "correlation_5"}
    %18 = ADORA.LocalMemAlloc memref<28x28xf32>  {Id = "2", KernelName = "correlation_5"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 27 {
        affine.store %cst_0, %18[%arg5, %arg5] : memref<28x28xf32>
        affine.for %arg6 = 0 to #map(%arg5) {
          %19 = affine.for %arg7 = 0 to 32 iter_args(%arg8 = %cst_1) -> (f32) {
            %20 = affine.load %16[%arg7, %arg5] : memref<32x27xf32>
            %21 = affine.load %17[%arg7, %arg5 + %arg6] : memref<32x27xf32>
            %22 = arith.mulf %20, %21 : f32
            %23 = arith.addf %arg8, %22 : f32
            affine.yield %23 : f32
          }
          affine.store %19, %18[%arg5, %arg5 + %arg6 + 1] : memref<28x28xf32>
          affine.store %19, %18[%arg5 + %arg6 + 1, %arg5] : memref<28x28xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "correlation_5"}
    ADORA.BlockStore %18, %arg2 [0, 0] : memref<28x28xf32> -> memref<?x28xf32>  {Id = "2", KernelName = "correlation_5"}
    affine.store %cst_0, %arg2[27, 27] : memref<?x28xf32>
    return
  }
}

