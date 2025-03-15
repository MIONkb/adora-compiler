#map = affine_map<(d0) -> (d0 + 8)>
#map1 = affine_map<(d0) -> (d0 + 16)>
#map2 = affine_map<(d0) -> (d0 + 24)>
#map3 = affine_map<(d0) -> (d0 + 32)>
#map4 = affine_map<(d0) -> (d0 + 40)>
#map5 = affine_map<(d0) -> (d0 + 48)>
#map6 = affine_map<(d0) -> (d0 + 56)>
module {
  func.func @forward_kernel_0(%arg0: memref<64x128xf32>, %arg1: memref<128x64xf32>, %arg2: memref<64x64xf32>) attributes {Kernel, forward_kernel_0} {
    %c0 = arith.constant 0 : index
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    affine.for %arg3 = 0 to 64 {
      %0 = ADORA.BlockLoad %arg2 [%arg3, %c0] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      %1 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      %2 = ADORA.BlockLoad %arg1 [0, %c0] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "2", KernelName = "forward_kernel_0"}
      %3 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      %4 = affine.apply #map(%c0)
      %5 = ADORA.BlockLoad %arg2 [%arg3, %4] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      %6 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      %7 = ADORA.BlockLoad %arg1 [0, %4] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "2", KernelName = "forward_kernel_0"}
      %8 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      %9 = affine.apply #map1(%c0)
      %10 = ADORA.BlockLoad %arg2 [%arg3, %9] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      %11 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      %12 = ADORA.BlockLoad %arg1 [0, %9] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "2", KernelName = "forward_kernel_0"}
      %13 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      %14 = affine.apply #map2(%c0)
      %15 = ADORA.BlockLoad %arg2 [%arg3, %14] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      %16 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      %17 = ADORA.BlockLoad %arg1 [0, %14] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "2", KernelName = "forward_kernel_0"}
      %18 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      %19 = affine.apply #map3(%c0)
      %20 = ADORA.BlockLoad %arg2 [%arg3, %19] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      %21 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      %22 = ADORA.BlockLoad %arg1 [0, %19] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "2", KernelName = "forward_kernel_0"}
      %23 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      %24 = affine.apply #map4(%c0)
      %25 = ADORA.BlockLoad %arg2 [%arg3, %24] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      %26 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      %27 = ADORA.BlockLoad %arg1 [0, %24] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "2", KernelName = "forward_kernel_0"}
      %28 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      %29 = affine.apply #map5(%c0)
      %30 = ADORA.BlockLoad %arg2 [%arg3, %29] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      %31 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      %32 = ADORA.BlockLoad %arg1 [0, %29] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "2", KernelName = "forward_kernel_0"}
      %33 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      %34 = affine.apply #map6(%c0)
      %35 = ADORA.BlockLoad %arg2 [%arg3, %34] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      %36 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      %37 = ADORA.BlockLoad %arg1 [0, %34] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "2", KernelName = "forward_kernel_0"}
      %38 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      ADORA.kernel {
        affine.for %arg4 = 0 to 8 {
          %46 = affine.load %0[0, %arg4] : memref<1x8xf32>
          %47 = affine.apply #map(%c0)
          %48 = affine.load %5[0, %arg4] : memref<1x8xf32>
          %49 = affine.apply #map1(%c0)
          %50 = affine.load %10[0, %arg4] : memref<1x8xf32>
          %51 = affine.apply #map2(%c0)
          %52 = affine.load %15[0, %arg4] : memref<1x8xf32>
          %53 = affine.apply #map3(%c0)
          %54 = affine.load %20[0, %arg4] : memref<1x8xf32>
          %55 = affine.apply #map4(%c0)
          %56 = affine.load %25[0, %arg4] : memref<1x8xf32>
          %57 = affine.apply #map5(%c0)
          %58 = affine.load %30[0, %arg4] : memref<1x8xf32>
          %59 = affine.apply #map6(%c0)
          %60 = affine.load %35[0, %arg4] : memref<1x8xf32>
          %61:8 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %46, %arg7 = %48, %arg8 = %50, %arg9 = %52, %arg10 = %54, %arg11 = %56, %arg12 = %58, %arg13 = %60) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
            %69 = affine.load %1[0, %arg5] : memref<1x128xf32>
            %70 = affine.load %2[%arg5, %arg4] : memref<128x8xf32>
            %71 = arith.mulf %69, %70 : f32
            %72 = arith.addf %arg6, %71 : f32
            %73 = affine.apply #map(%c0)
            %74 = affine.load %6[0, %arg5] : memref<1x128xf32>
            %75 = affine.load %7[%arg5, %arg4] : memref<128x8xf32>
            %76 = arith.mulf %74, %75 : f32
            %77 = arith.addf %arg7, %76 : f32
            %78 = affine.apply #map1(%c0)
            %79 = affine.load %11[0, %arg5] : memref<1x128xf32>
            %80 = affine.load %12[%arg5, %arg4] : memref<128x8xf32>
            %81 = arith.mulf %79, %80 : f32
            %82 = arith.addf %arg8, %81 : f32
            %83 = affine.apply #map2(%c0)
            %84 = affine.load %16[0, %arg5] : memref<1x128xf32>
            %85 = affine.load %17[%arg5, %arg4] : memref<128x8xf32>
            %86 = arith.mulf %84, %85 : f32
            %87 = arith.addf %arg9, %86 : f32
            %88 = affine.apply #map3(%c0)
            %89 = affine.load %21[0, %arg5] : memref<1x128xf32>
            %90 = affine.load %22[%arg5, %arg4] : memref<128x8xf32>
            %91 = arith.mulf %89, %90 : f32
            %92 = arith.addf %arg10, %91 : f32
            %93 = affine.apply #map4(%c0)
            %94 = affine.load %26[0, %arg5] : memref<1x128xf32>
            %95 = affine.load %27[%arg5, %arg4] : memref<128x8xf32>
            %96 = arith.mulf %94, %95 : f32
            %97 = arith.addf %arg11, %96 : f32
            %98 = affine.apply #map5(%c0)
            %99 = affine.load %31[0, %arg5] : memref<1x128xf32>
            %100 = affine.load %32[%arg5, %arg4] : memref<128x8xf32>
            %101 = arith.mulf %99, %100 : f32
            %102 = arith.addf %arg12, %101 : f32
            %103 = affine.apply #map6(%c0)
            %104 = affine.load %36[0, %arg5] : memref<1x128xf32>
            %105 = affine.load %37[%arg5, %arg4] : memref<128x8xf32>
            %106 = arith.mulf %104, %105 : f32
            %107 = arith.addf %arg13, %106 : f32
            affine.yield %72, %77, %82, %87, %92, %97, %102, %107 : f32, f32, f32, f32, f32, f32, f32, f32
          }
          affine.store %61#0, %3[0, %arg4] : memref<1x8xf32>
          %62 = affine.apply #map(%c0)
          affine.store %61#1, %8[0, %arg4] : memref<1x8xf32>
          %63 = affine.apply #map1(%c0)
          affine.store %61#2, %13[0, %arg4] : memref<1x8xf32>
          %64 = affine.apply #map2(%c0)
          affine.store %61#3, %18[0, %arg4] : memref<1x8xf32>
          %65 = affine.apply #map3(%c0)
          affine.store %61#4, %23[0, %arg4] : memref<1x8xf32>
          %66 = affine.apply #map4(%c0)
          affine.store %61#5, %28[0, %arg4] : memref<1x8xf32>
          %67 = affine.apply #map5(%c0)
          affine.store %61#6, %33[0, %arg4] : memref<1x8xf32>
          %68 = affine.apply #map6(%c0)
          affine.store %61#7, %38[0, %arg4] : memref<1x8xf32>
        }
        ADORA.terminator
      } {KernelName = "forward_kernel_0"}
      ADORA.BlockStore %3, %arg2 [%arg3, %c0] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      %39 = affine.apply #map(%c0)
      ADORA.BlockStore %8, %arg2 [%arg3, %39] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      %40 = affine.apply #map1(%c0)
      ADORA.BlockStore %13, %arg2 [%arg3, %40] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      %41 = affine.apply #map2(%c0)
      ADORA.BlockStore %18, %arg2 [%arg3, %41] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      %42 = affine.apply #map3(%c0)
      ADORA.BlockStore %23, %arg2 [%arg3, %42] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      %43 = affine.apply #map4(%c0)
      ADORA.BlockStore %28, %arg2 [%arg3, %43] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      %44 = affine.apply #map5(%c0)
      ADORA.BlockStore %33, %arg2 [%arg3, %44] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      %45 = affine.apply #map6(%c0)
      ADORA.BlockStore %38, %arg2 [%arg3, %45] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
    }
    return
  }
}
