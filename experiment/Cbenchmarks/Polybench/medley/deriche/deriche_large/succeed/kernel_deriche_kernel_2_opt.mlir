#map = affine_map<(d0) -> (d0 + 1080)>
module {
  func.func @deriche_kernel_2(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>) attributes {Kernel, deriche_kernel_2} {
    %c0 = arith.constant 0 : index
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    affine.for %arg3 = 0 to 4096 {
      %0 = ADORA.BlockLoad %arg0 [%arg3, %c0] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "0", KernelName = "deriche_kernel_2"}
      %1 = ADORA.BlockLoad %arg1 [%arg3, %c0] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "1", KernelName = "deriche_kernel_2"}
      %2 = ADORA.LocalMemAlloc memref<1x1080xf32>  {Id = "2", KernelName = "deriche_kernel_2"}
      %3 = affine.apply #map(%c0)
      %4 = ADORA.BlockLoad %arg0 [%arg3, %3] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "3", KernelName = "deriche_kernel_2"}
      %5 = ADORA.BlockLoad %arg1 [%arg3, %3] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "4", KernelName = "deriche_kernel_2"}
      %6 = ADORA.LocalMemAlloc memref<1x1080xf32>  {Id = "5", KernelName = "deriche_kernel_2"}
      ADORA.kernel {
        affine.for %arg4 = 0 to 1080 {
          %8 = affine.load %0[0, %arg4] : memref<1x1080xf32>
          %9 = affine.load %1[0, %arg4] : memref<1x1080xf32>
          %10 = arith.addf %8, %9 : f32
          affine.store %10, %2[0, %arg4] : memref<1x1080xf32>
          %11 = affine.apply #map(%c0)
          %12 = affine.load %4[0, %arg4] : memref<1x1080xf32>
          %13 = affine.load %5[0, %arg4] : memref<1x1080xf32>
          %14 = arith.addf %12, %13 : f32
          affine.store %14, %6[0, %arg4] : memref<1x1080xf32>
        }
        ADORA.terminator
      } {KernelName = "deriche_kernel_2"}
      ADORA.BlockStore %2, %arg2 [%arg3, %c0] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "2", KernelName = "deriche_kernel_2"}
      %7 = affine.apply #map(%c0)
      ADORA.BlockStore %6, %arg2 [%arg3, %7] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "5", KernelName = "deriche_kernel_2"}
    }
    return
  }
}