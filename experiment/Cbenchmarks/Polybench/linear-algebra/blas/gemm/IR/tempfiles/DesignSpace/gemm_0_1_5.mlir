#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
#map3 = affine_map<(d0) -> (d0 + 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm(%arg0: f32, %arg1: f32, %arg2: memref<?x25xf32>, %arg3: memref<?x30xf32>, %arg4: memref<?x25xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg5 = 0 to 20 step 5 {
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
      %11 = affine.apply #map3(%arg5)
      %12 = ADORA.BlockLoad %arg2 [%11, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
      %13 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      ADORA.kernel {
        affine.for %arg6 = 0 to 25 {
          %42 = affine.load %0[0, %arg6] : memref<1x25xf32>
          %43 = arith.mulf %42, %arg1 : f32
          affine.store %43, %1[0, %arg6] : memref<1x25xf32>
          %44 = affine.load %3[0, %arg6] : memref<1x25xf32>
          %45 = arith.mulf %44, %arg1 : f32
          affine.store %45, %4[0, %arg6] : memref<1x25xf32>
          %46 = affine.load %6[0, %arg6] : memref<1x25xf32>
          %47 = arith.mulf %46, %arg1 : f32
          affine.store %47, %7[0, %arg6] : memref<1x25xf32>
          %48 = affine.load %9[0, %arg6] : memref<1x25xf32>
          %49 = arith.mulf %48, %arg1 : f32
          affine.store %49, %10[0, %arg6] : memref<1x25xf32>
          %50 = affine.load %12[0, %arg6] : memref<1x25xf32>
          %51 = arith.mulf %50, %arg1 : f32
          affine.store %51, %13[0, %arg6] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "gemm_0"}
      ADORA.BlockStore %1, %arg2 [%arg5, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %14 = ADORA.BlockLoad %arg3 [%arg5, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %15 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %16 = ADORA.BlockLoad %arg2 [%arg5, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %17 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %18 = affine.apply #map(%arg5)
      ADORA.BlockStore %4, %arg2 [%18, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %19 = ADORA.BlockLoad %arg3 [%18, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %20 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %21 = ADORA.BlockLoad %arg2 [%18, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %22 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %23 = affine.apply #map1(%arg5)
      ADORA.BlockStore %7, %arg2 [%23, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %24 = ADORA.BlockLoad %arg3 [%23, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %25 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %26 = ADORA.BlockLoad %arg2 [%23, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %27 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %28 = affine.apply #map2(%arg5)
      ADORA.BlockStore %10, %arg2 [%28, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %29 = ADORA.BlockLoad %arg3 [%28, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %30 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %31 = ADORA.BlockLoad %arg2 [%28, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %32 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %33 = affine.apply #map3(%arg5)
      ADORA.BlockStore %13, %arg2 [%33, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %34 = ADORA.BlockLoad %arg3 [%33, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "0", KernelName = "gemm_1"}
      %35 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "1", KernelName = "gemm_1"}
      %36 = ADORA.BlockLoad %arg2 [%33, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "2", KernelName = "gemm_1"}
      %37 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      ADORA.kernel {
        affine.for %arg6 = 0 to 30 {
          affine.for %arg7 = 0 to 25 {
            %42 = affine.load %14[0, %arg6] : memref<1x30xf32>
            %43 = arith.mulf %arg0, %42 : f32
            %44 = affine.load %15[%arg6, %arg7] : memref<30x25xf32>
            %45 = arith.mulf %43, %44 : f32
            %46 = affine.load %16[0, %arg7] : memref<1x25xf32>
            %47 = arith.addf %46, %45 : f32
            affine.store %47, %17[0, %arg7] : memref<1x25xf32>
            %48 = affine.apply #map(%arg5)
            %49 = affine.load %19[0, %arg6] : memref<1x30xf32>
            %50 = arith.mulf %arg0, %49 : f32
            %51 = affine.load %20[%arg6, %arg7] : memref<30x25xf32>
            %52 = arith.mulf %50, %51 : f32
            %53 = affine.load %21[0, %arg7] : memref<1x25xf32>
            %54 = arith.addf %53, %52 : f32
            affine.store %54, %22[0, %arg7] : memref<1x25xf32>
            %55 = affine.apply #map1(%arg5)
            %56 = affine.load %24[0, %arg6] : memref<1x30xf32>
            %57 = arith.mulf %arg0, %56 : f32
            %58 = affine.load %25[%arg6, %arg7] : memref<30x25xf32>
            %59 = arith.mulf %57, %58 : f32
            %60 = affine.load %26[0, %arg7] : memref<1x25xf32>
            %61 = arith.addf %60, %59 : f32
            affine.store %61, %27[0, %arg7] : memref<1x25xf32>
            %62 = affine.apply #map2(%arg5)
            %63 = affine.load %29[0, %arg6] : memref<1x30xf32>
            %64 = arith.mulf %arg0, %63 : f32
            %65 = affine.load %30[%arg6, %arg7] : memref<30x25xf32>
            %66 = arith.mulf %64, %65 : f32
            %67 = affine.load %31[0, %arg7] : memref<1x25xf32>
            %68 = arith.addf %67, %66 : f32
            affine.store %68, %32[0, %arg7] : memref<1x25xf32>
            %69 = affine.apply #map3(%arg5)
            %70 = affine.load %34[0, %arg6] : memref<1x30xf32>
            %71 = arith.mulf %arg0, %70 : f32
            %72 = affine.load %35[%arg6, %arg7] : memref<30x25xf32>
            %73 = arith.mulf %71, %72 : f32
            %74 = affine.load %36[0, %arg7] : memref<1x25xf32>
            %75 = arith.addf %74, %73 : f32
            affine.store %75, %37[0, %arg7] : memref<1x25xf32>
          }
        }
        ADORA.terminator
      } {KernelName = "gemm_1"}
      ADORA.BlockStore %17, %arg2 [%arg5, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %38 = affine.apply #map(%arg5)
      ADORA.BlockStore %22, %arg2 [%38, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %39 = affine.apply #map1(%arg5)
      ADORA.BlockStore %27, %arg2 [%39, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %40 = affine.apply #map2(%arg5)
      ADORA.BlockStore %32, %arg2 [%40, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %41 = affine.apply #map3(%arg5)
      ADORA.BlockStore %37, %arg2 [%41, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"}
    }
    return
  }
}
