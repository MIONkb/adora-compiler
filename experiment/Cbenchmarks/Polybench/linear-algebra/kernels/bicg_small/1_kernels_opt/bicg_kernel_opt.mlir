module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @bicg(%arg0: memref<124x116xf32>, %arg1: memref<116xf32>, %arg2: memref<124xf32>, %arg3: memref<116xf32>, %arg4: memref<124xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg5 = 0 to 116 {
      affine.store %cst, %arg1[%arg5] : memref<116xf32>
    }
    affine.for %arg5 = 0 to 124 step 4 {
      %0 = ADORA.BlockLoad %arg1 [0] : memref<116xf32> -> memref<116xf32>  {Id = "0", KernelName = ""}
      %2 = ADORA.BlockLoad %arg4 [%arg5] : memref<124xf32> -> memref<4xf32>  {Id = "2", KernelName = ""}
      %3 = ADORA.BlockLoad %arg0 [%arg5, 0] : memref<124x116xf32> -> memref<4x116xf32>  {Id = "3", KernelName = ""}
      %4 = ADORA.BlockLoad %arg2 [%arg5] : memref<124xf32> -> memref<4xf32>  {Id = "4", KernelName = ""}
      %5 = ADORA.BlockLoad %arg3 [0] : memref<116xf32> -> memref<116xf32>  {Id = "5", KernelName = ""}
      %6 = ADORA.LocalMemAlloc memref<4xf32>  {Id = "6", KernelName = ""}
      ADORA.kernel {
        affine.for %arg6 = 0 to 4 {
        //   affine.store %cst, %6[%arg6] : memref<4xf32>
          %16 =  affine.for %arg7 = 0 to 116 iter_args(%arg8 = %cst) -> (f32) {
            %7 = affine.load %0[%arg7] : memref<116xf32>
            %8 = affine.load %2[%arg6] : memref<4xf32>
            %9 = affine.load %3[%arg6, %arg7] : memref<4x116xf32>
            %10 = arith.mulf %8, %9 : f32
            %11 = arith.addf %7, %10 : f32
            // affine.store %11, %0[%arg7] : memref<116xf32>
            // %12 = affine.load %4[%arg6] : memref<4xf32>
            %13 = affine.load %5[%arg7] : memref<116xf32>
            %14 = arith.mulf %9, %13 : f32
            %15 = arith.addf %arg8, %14 : f32
            // affine.store %15, %6[%arg6] : memref<4xf32>
            affine.yield %15 : f32
          }
          affine.store %16, %6[%arg6] : memref<4xf32>
        }
        ADORA.terminator
      }
      ADORA.BlockStore %6, %arg2 [%arg5] : memref<4xf32> -> memref<124xf32>  {Id = "6", KernelName = ""}
      ADORA.BlockStore %0, %arg1 [0] : memref<116xf32> -> memref<116xf32>  {Id = "0", KernelName = ""}
    }
    return
  }
}