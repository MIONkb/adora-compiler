module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm(%arg0: f32, %arg1: f32, %arg2: memref<?x25xf32>, %arg3: memref<?x30xf32>, %arg4: memref<?x25xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg5 = 0 to 20 {
      %0 = ADORA.BlockLoad %arg2 [%arg5, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %1 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      ADORA.kernel {
        affine.for %arg6 = 0 to 25 {
          %6 = affine.load %0[0, %arg6] : memref<1x25xf32>
          %7 = arith.mulf %6, %arg1 : f32
          affine.store %7, %1[0, %arg6] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "gemm_0"}
      ADORA.BlockStore %1, %arg2 [%arg5, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %2 = ADORA.BlockLoad %arg3 [%arg5, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %3 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %4 = ADORA.BlockLoad %arg2 [%arg5, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %5 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      ADORA.kernel {
        affine.for %arg6 = 0 to 30 {
          affine.for %arg7 = 0 to 25 {
            %6 = affine.load %2[0, %arg6] : memref<1x30xf32>
            %7 = arith.mulf %arg0, %6 : f32
            %8 = affine.load %3[%arg6, %arg7] : memref<30x25xf32>
            %9 = arith.mulf %7, %8 : f32
            %10 = affine.load %4[0, %arg7] : memref<1x25xf32>
            %11 = arith.addf %10, %9 : f32
            affine.store %11, %5[0, %arg7] : memref<1x25xf32>
          }
        }
        ADORA.terminator
      } {KernelName = "gemm_1"}
      ADORA.BlockStore %5, %arg2 [%arg5, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
    }
    return
  }
}
