#map = affine_map<(d0) -> (d0 + 8)>
module {
  func.func @forward_kernel_0(%arg0: memref<64x128xf32>, %arg1: memref<128x64xf32>, %arg2: memref<64x64xf32>) attributes {Kernel, forward_kernel_0} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 step 16 {
        %0 = ADORA.BlockLoad %arg2 [%arg3, %arg4] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "0", KernelName = "forward_kernel_0"}
        %1 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "1", KernelName = "forward_kernel_0"}
        %2 = ADORA.BlockLoad %arg1 [0, %arg4] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "2", KernelName = "forward_kernel_0"}
        %3 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "3", KernelName = "forward_kernel_0"}
        %4 = affine.apply #map(%arg4)
        %5 = ADORA.BlockLoad %arg2 [%arg3, %4] : memref<64x64xf32> -> memref<1x8xf32>  {Id = "0", KernelName = "forward_kernel_0"}
        %6 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x128xf32> -> memref<1x128xf32>  {Id = "1", KernelName = "forward_kernel_0"}
        %7 = ADORA.BlockLoad %arg1 [0, %4] : memref<128x64xf32> -> memref<128x8xf32>  {Id = "2", KernelName = "forward_kernel_0"}
        %8 = ADORA.LocalMemAlloc memref<1x8xf32>  {Id = "3", KernelName = "forward_kernel_0"}
        ADORA.kernel {
          affine.for %arg5 = 0 to 8 {
            %10 = affine.load %0[0, %arg5] : memref<1x8xf32>
            %11 = affine.apply #map(%arg4)
            %12 = affine.load %5[0, %arg5] : memref<1x8xf32>
            %13:2 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %10, %arg8 = %12) -> (f32, f32) {
              %15 = affine.load %1[0, %arg6] : memref<1x128xf32>
              %16 = affine.load %2[%arg6, %arg5] : memref<128x8xf32>
              %17 = arith.mulf %15, %16 : f32
              %18 = arith.addf %arg7, %17 : f32
              %19 = affine.apply #map(%arg4)
              %20 = affine.load %6[0, %arg6] : memref<1x128xf32>
              %21 = affine.load %7[%arg6, %arg5] : memref<128x8xf32>
              %22 = arith.mulf %20, %21 : f32
              %23 = arith.addf %arg8, %22 : f32
              affine.yield %18, %23 : f32, f32
            }
            affine.store %13#0, %3[0, %arg5] : memref<1x8xf32>
            %14 = affine.apply #map(%arg4)
            affine.store %13#1, %8[0, %arg5] : memref<1x8xf32>
          }
          ADORA.terminator
        } {KernelName = "forward_kernel_0"}
        ADORA.BlockStore %3, %arg2 [%arg3, %arg4] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
        %9 = affine.apply #map(%arg4)
        ADORA.BlockStore %8, %arg2 [%arg3, %9] : memref<1x8xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "forward_kernel_0"}
      }
    }
    return
  }
}
