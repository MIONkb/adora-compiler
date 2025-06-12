#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm(%arg0: f32, %arg1: f32, %arg2: memref<?x25xf32>, %arg3: memref<?x30xf32>, %arg4: memref<?x25xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg5 = 0 to 20 step 4 {
      %0 = ADORA.BlockLoad %arg2 [%arg5, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %1 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %2 = affine.apply #map(%arg5)
      %3 = ADORA.BlockLoad %arg2 [%2, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %4 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %5 = affine.apply #map1(%arg5)
      %6 = ADORA.BlockLoad %arg2 [%5, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %7 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %8 = affine.apply #map2(%arg5)
      %9 = ADORA.BlockLoad %arg2 [%8, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %10 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      ADORA.kernel {
        affine.for %arg6 = 0 to 25 {
          %33 = affine.load %0[0, %arg6] : memref<1x25xf32>
          %34 = arith.mulf %33, %arg1 : f32
          affine.store %34, %1[0, %arg6] : memref<1x25xf32>
          %35 = affine.load %3[0, %arg6] : memref<1x25xf32>
          %36 = arith.mulf %35, %arg1 : f32
          affine.store %36, %4[0, %arg6] : memref<1x25xf32>
          %37 = affine.load %6[0, %arg6] : memref<1x25xf32>
          %38 = arith.mulf %37, %arg1 : f32
          affine.store %38, %7[0, %arg6] : memref<1x25xf32>
          %39 = affine.load %9[0, %arg6] : memref<1x25xf32>
          %40 = arith.mulf %39, %arg1 : f32
          affine.store %40, %10[0, %arg6] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "gemm_0"}
      ADORA.BlockStore %1, %arg2 [%arg5, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %11 = ADORA.BlockLoad %arg3 [%arg5, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %12 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %13 = ADORA.BlockLoad %arg2 [%arg5, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %14 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %15 = affine.apply #map(%arg5)
      ADORA.BlockStore %4, %arg2 [%15, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %16 = ADORA.BlockLoad %arg3 [%15, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %17 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %18 = ADORA.BlockLoad %arg2 [%15, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %19 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %20 = affine.apply #map1(%arg5)
      ADORA.BlockStore %7, %arg2 [%20, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %21 = ADORA.BlockLoad %arg3 [%20, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %22 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %23 = ADORA.BlockLoad %arg2 [%20, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %24 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %25 = affine.apply #map2(%arg5)
      ADORA.BlockStore %10, %arg2 [%25, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %26 = ADORA.BlockLoad %arg3 [%25, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %27 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %28 = ADORA.BlockLoad %arg2 [%25, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %29 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      ADORA.kernel {
        affine.for %arg6 = 0 to 30 {
          affine.for %arg7 = 0 to 25 {
            %33 = affine.load %11[0, %arg6] : memref<1x30xf32>
            %34 = arith.mulf %arg0, %33 : f32
            %35 = affine.load %12[%arg6, %arg7] : memref<30x25xf32>
            %36 = arith.mulf %34, %35 : f32
            %37 = affine.load %13[0, %arg7] : memref<1x25xf32>
            %38 = arith.addf %37, %36 : f32
            affine.store %38, %14[0, %arg7] : memref<1x25xf32>
            %39 = affine.apply #map(%arg5)
            %40 = affine.load %16[0, %arg6] : memref<1x30xf32>
            %41 = arith.mulf %arg0, %40 : f32
            %42 = affine.load %17[%arg6, %arg7] : memref<30x25xf32>
            %43 = arith.mulf %41, %42 : f32
            %44 = affine.load %18[0, %arg7] : memref<1x25xf32>
            %45 = arith.addf %44, %43 : f32
            affine.store %45, %19[0, %arg7] : memref<1x25xf32>
            %46 = affine.apply #map1(%arg5)
            %47 = affine.load %21[0, %arg6] : memref<1x30xf32>
            %48 = arith.mulf %arg0, %47 : f32
            %49 = affine.load %22[%arg6, %arg7] : memref<30x25xf32>
            %50 = arith.mulf %48, %49 : f32
            %51 = affine.load %23[0, %arg7] : memref<1x25xf32>
            %52 = arith.addf %51, %50 : f32
            affine.store %52, %24[0, %arg7] : memref<1x25xf32>
            %53 = affine.apply #map2(%arg5)
            %54 = affine.load %26[0, %arg6] : memref<1x30xf32>
            %55 = arith.mulf %arg0, %54 : f32
            %56 = affine.load %27[%arg6, %arg7] : memref<30x25xf32>
            %57 = arith.mulf %55, %56 : f32
            %58 = affine.load %28[0, %arg7] : memref<1x25xf32>
            %59 = arith.addf %58, %57 : f32
            affine.store %59, %29[0, %arg7] : memref<1x25xf32>
          }
        }
        ADORA.terminator
      } {KernelName = "gemm_1"}
      ADORA.BlockStore %14, %arg2 [%arg5, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %30 = affine.apply #map(%arg5)
      ADORA.BlockStore %19, %arg2 [%30, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %31 = affine.apply #map1(%arg5)
      ADORA.BlockStore %24, %arg2 [%31, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %32 = affine.apply #map2(%arg5)
      ADORA.BlockStore %29, %arg2 [%32, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
    }
    return
  }
}
