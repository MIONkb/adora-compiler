#map = affine_map<(d0) -> (d0 + 32)>
#map1 = affine_map<(d0) -> (d0 + 64)>
#map2 = affine_map<(d0) -> (d0 + 96)>
module {
  func.func @forward_kernel_0(%arg0: memref<1x128x64xf32>, %arg1: memref<1x128x64xf32>) attributes {Kernel, forward_kernel_0} {
    %c0 = arith.constant 0 : index
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 1 {
      %0 = ADORA.BlockLoad %arg0 [0, %c0, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      %1 = ADORA.LocalMemAlloc memref<1x32x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      %2 = affine.apply #map(%c0)
      %3 = ADORA.BlockLoad %arg0 [0, %2, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      %4 = ADORA.LocalMemAlloc memref<1x32x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      %5 = affine.apply #map1(%c0)
      %6 = ADORA.BlockLoad %arg0 [0, %5, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      %7 = ADORA.LocalMemAlloc memref<1x32x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      %8 = affine.apply #map2(%c0)
      %9 = ADORA.BlockLoad %arg0 [0, %8, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "0", KernelName = "forward_kernel_0"}
      %10 = ADORA.LocalMemAlloc memref<1x32x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      ADORA.kernel {
        affine.for %arg3 = 0 to 32 {
          affine.for %arg4 = 0 to 64 {
            %14 = affine.load %0[0, %arg3, %arg4] : memref<1x32x64xf32>
            %15 = arith.cmpf ugt, %14, %cst : f32
            %16 = arith.select %15, %14, %cst : f32
            affine.store %16, %1[0, %arg3, %arg4] : memref<1x32x64xf32>
            %c32 = arith.constant 32 : index
            %17 = affine.load %3[0, %arg3, %arg4] : memref<1x32x64xf32>
            %18 = arith.cmpf ugt, %17, %cst : f32
            %19 = arith.select %18, %17, %cst : f32
            affine.store %19, %4[0, %arg3, %arg4] : memref<1x32x64xf32>
            %c64 = arith.constant 64 : index
            %20 = affine.load %6[0, %arg3, %arg4] : memref<1x32x64xf32>
            %21 = arith.cmpf ugt, %20, %cst : f32
            %22 = arith.select %21, %20, %cst : f32
            affine.store %22, %7[0, %arg3, %arg4] : memref<1x32x64xf32>
            %c96 = arith.constant 96 : index
            %23 = affine.load %9[0, %arg3, %arg4] : memref<1x32x64xf32>
            %24 = arith.cmpf ugt, %23, %cst : f32
            %25 = arith.select %24, %23, %cst : f32
            affine.store %25, %10[0, %arg3, %arg4] : memref<1x32x64xf32>
          }
        }
        ADORA.terminator
      } {KernelName = "forward_kernel_0"}
      ADORA.BlockStore %1, %arg1 [0, %c0, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      %11 = affine.apply #map(%c0)
      ADORA.BlockStore %4, %arg1 [0, %11, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      %12 = affine.apply #map1(%c0)
      ADORA.BlockStore %7, %arg1 [0, %12, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      %13 = affine.apply #map2(%c0)
      ADORA.BlockStore %10, %arg1 [0, %13, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
    }
    return
  }
}
