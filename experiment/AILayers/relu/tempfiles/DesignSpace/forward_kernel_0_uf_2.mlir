#map = affine_map<(d0) -> (d0 + 32)>
module {
  func.func @forward_kernel_0(%arg0: memref<1x128x64xf32>, %arg1: memref<1x128x64xf32>) attributes {Kernel, forward_kernel_0} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 step 64 {
        %0 = ADORA.BlockLoad %arg0 [0, %arg3, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "0", KernelName = "forward_kernel_0"}
        %1 = ADORA.LocalMemAlloc memref<1x32x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
        %2 = affine.apply #map(%arg3)
        %3 = ADORA.BlockLoad %arg0 [0, %2, 0] : memref<1x128x64xf32> -> memref<1x32x64xf32>  {Id = "0", KernelName = "forward_kernel_0"}
        %4 = ADORA.LocalMemAlloc memref<1x32x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
        ADORA.kernel {
          affine.for %arg4 = 0 to 32 {
            affine.for %arg5 = 0 to 64 {
              %6 = affine.load %0[0, %arg4, %arg5] : memref<1x32x64xf32>
              %7 = arith.cmpf ugt, %6, %cst : f32
              %8 = arith.select %7, %6, %cst : f32
              affine.store %8, %1[0, %arg4, %arg5] : memref<1x32x64xf32>
              %9 = affine.apply #map(%arg3)
              %10 = affine.load %3[0, %arg4, %arg5] : memref<1x32x64xf32>
              %11 = arith.cmpf ugt, %10, %cst : f32
              %12 = arith.select %11, %10, %cst : f32
              affine.store %12, %4[0, %arg4, %arg5] : memref<1x32x64xf32>
            }
          }
          ADORA.terminator
        } {KernelName = "forward_kernel_0"}
        ADORA.BlockStore %1, %arg1 [0, %arg3, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
        %5 = affine.apply #map(%arg3)
        ADORA.BlockStore %4, %arg1 [0, %5, 0] : memref<1x32x64xf32> -> memref<1x128x64xf32>  {Id = "1", KernelName = "forward_kernel_0"}
      }
    }
    return
  }
}
