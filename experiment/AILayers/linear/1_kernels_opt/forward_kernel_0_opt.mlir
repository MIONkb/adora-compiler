#map = affine_map<(d0) -> (d0 + 8)>
#map1 = affine_map<(d0) -> (d0 + 16)>
#map2 = affine_map<(d0) -> (d0 + 24)>
module {
  func.func @forward_kernel_0(%arg0: memref<64x128xf32>, %arg1: memref<128x64xf32>, %arg2: memref<64x64xf32>) attributes {Kernel, forward_kernel_0} {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 step 32 {
        %0 = ADORA.BlockLoad %arg2 [%arg3, %arg4] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "0", KernelName = "forward_kernel_0"}
        %1 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "1", KernelName = "forward_kernel_0"}
        %2 = ADORA.BlockLoad %arg1 [0, %arg4] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "2", KernelName = "forward_kernel_0"}
        %3 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "3", KernelName = "forward_kernel_0"}
        %4 = affine.apply #map(%arg4)
        %5 = ADORA.BlockLoad %arg2 [%arg3, %4] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "4", KernelName = "forward_kernel_0"}
        %6 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "5", KernelName = "forward_kernel_0"}
        %7 = ADORA.BlockLoad %arg1 [0, %4] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "6", KernelName = "forward_kernel_0"}
        %8 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "7", KernelName = "forward_kernel_0"}
        %9 = affine.apply #map1(%arg4)
        %10 = ADORA.BlockLoad %arg2 [%arg3, %9] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "8", KernelName = "forward_kernel_0"}
        %11 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "9", KernelName = "forward_kernel_0"}
        %12 = ADORA.BlockLoad %arg1 [0, %9] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "10", KernelName = "forward_kernel_0"}
        %13 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "11", KernelName = "forward_kernel_0"}
        %14 = affine.apply #map2(%arg4)
        %15 = ADORA.BlockLoad %arg2 [%arg3, %14] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "12", KernelName = "forward_kernel_0"}
        %16 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "13", KernelName = "forward_kernel_0"}
        %17 = ADORA.BlockLoad %arg1 [0, %14] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "14", KernelName = "forward_kernel_0"}
        %18 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "15", KernelName = "forward_kernel_0"}
        ADORA.kernel {
          affine.for %arg5 = 0 to 8 {
            %22 = affine.load %0[0, %arg5] : memref<1x8xf32>
            %23 = affine.load %5[0, %arg5] : memref<1x8xf32>
            %24 = affine.load %10[0, %arg5] : memref<1x8xf32>
            %25 = affine.load %15[0, %arg5] : memref<1x8xf32>
            %26:4 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %22, %arg8 = %23, %arg9 = %24, %arg10 = %25) -> (f32, f32, f32, f32) {
              %27 = affine.load %1[0, %arg6] : memref<1x128xf32>
              %28 = affine.load %2[%arg6, %arg5] : memref<128x8xf32>
              %29 = arith.mulf %27, %28 : f32
              %30 = arith.addf %arg7, %29 : f32
              %31 = affine.load %6[0, %arg6] : memref<1x128xf32>
              %32 = affine.load %7[%arg6, %arg5] : memref<128x8xf32>
              %33 = arith.mulf %31, %32 : f32
              %34 = arith.addf %arg8, %33 : f32
              %35 = affine.load %11[0, %arg6] : memref<1x128xf32>
              %36 = affine.load %12[%arg6, %arg5] : memref<128x8xf32>
              %37 = arith.mulf %35, %36 : f32
              %38 = arith.addf %arg9, %37 : f32
              %39 = affine.load %16[0, %arg6] : memref<1x128xf32>
              %40 = affine.load %17[%arg6, %arg5] : memref<128x8xf32>
              %41 = arith.mulf %39, %40 : f32
              %42 = arith.addf %arg10, %41 : f32
              affine.yield %30, %34, %38, %42 : f32, f32, f32, f32
            }
            affine.store %26#0, %3[0, %arg5] : memref<1x8xf32>
            affine.store %26#1, %8[0, %arg5] : memref<1x8xf32>
            affine.store %26#2, %13[0, %arg5] : memref<1x8xf32>
            affine.store %26#3, %18[0, %arg5] : memref<1x8xf32>
          }
          ADORA.terminator
        } {KernelName = "forward_kernel_0"}
        ADORA.BlockStore %3, %arg2 [%arg3, %arg4] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
        %19 = affine.apply #map(%arg4)
        ADORA.BlockStore %8, %arg2 [%arg3, %19] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "7", KernelName = "forward_kernel_0"}
        %20 = affine.apply #map1(%arg4)
        ADORA.BlockStore %13, %arg2 [%arg3, %20] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "11", KernelName = "forward_kernel_0"}
        %21 = affine.apply #map2(%arg4)
        ADORA.BlockStore %18, %arg2 [%arg3, %21] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "15", KernelName = "forward_kernel_0"}
      }
    }
    return
  }
}