#map = affine_map<(d0) -> (d0 + 16)>
#map1 = affine_map<(d0) -> (d0 + 32)>
#map2 = affine_map<(d0) -> (d0 + 48)>
module {
  func.func @forward_kernel_1(%arg0: memref<64x64xf32>, %arg1: memref<64xf32>, %arg2: memref<64x64xf32>) attributes {Kernel, forward_kernel_1} {
    %c0 = arith.constant 0 : index
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = ADORA.BlockLoad %arg0 [%c0, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "0", KernelName = "forward_kernel_1"}
    %1 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "1", KernelName = "forward_kernel_1"}
    %2 = ADORA.LocalMemAlloc memref<16x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
    %3 = affine.apply #map(%c0)
    %4 = ADORA.BlockLoad %arg0 [%3, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "0", KernelName = "forward_kernel_1"}
    %5 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "1", KernelName = "forward_kernel_1"}
    %6 = ADORA.LocalMemAlloc memref<16x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
    %7 = affine.apply #map1(%c0)
    %8 = ADORA.BlockLoad %arg0 [%7, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "0", KernelName = "forward_kernel_1"}
    %9 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "1", KernelName = "forward_kernel_1"}
    %10 = ADORA.LocalMemAlloc memref<16x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
    %11 = affine.apply #map2(%c0)
    %12 = ADORA.BlockLoad %arg0 [%11, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "0", KernelName = "forward_kernel_1"}
    %13 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "1", KernelName = "forward_kernel_1"}
    %14 = ADORA.LocalMemAlloc memref<16x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 64 {
          %18 = affine.load %0[%arg3, %arg4] : memref<16x64xf32>
          %19 = affine.load %1[%arg4] : memref<64xf32>
          %20 = arith.addf %18, %19 : f32
          affine.store %20, %2[%arg3, %arg4] : memref<16x64xf32>
          %21 = affine.apply #map(%c0)
          %22 = affine.load %4[%arg3, %arg4] : memref<16x64xf32>
          %23 = affine.load %5[%arg4] : memref<64xf32>
          %24 = arith.addf %22, %23 : f32
          affine.store %24, %6[%arg3, %arg4] : memref<16x64xf32>
          %25 = affine.apply #map1(%c0)
          %26 = affine.load %8[%arg3, %arg4] : memref<16x64xf32>
          %27 = affine.load %9[%arg4] : memref<64xf32>
          %28 = arith.addf %26, %27 : f32
          affine.store %28, %10[%arg3, %arg4] : memref<16x64xf32>
          %29 = affine.apply #map2(%c0)
          %30 = affine.load %12[%arg3, %arg4] : memref<16x64xf32>
          %31 = affine.load %13[%arg4] : memref<64xf32>
          %32 = arith.addf %30, %31 : f32
          affine.store %32, %14[%arg3, %arg4] : memref<16x64xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "forward_kernel_1"}
    ADORA.BlockStore %2, %arg2 [%c0, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
    %15 = affine.apply #map(%c0)
    ADORA.BlockStore %6, %arg2 [%15, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
    %16 = affine.apply #map1(%c0)
    ADORA.BlockStore %10, %arg2 [%16, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
    %17 = affine.apply #map2(%c0)
    ADORA.BlockStore %14, %arg2 [%17, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
    return
  }
}
