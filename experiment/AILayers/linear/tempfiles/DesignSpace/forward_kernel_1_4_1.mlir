#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
module {
  func.func @forward_kernel_1(%arg0: memref<64x64xf32>, %arg1: memref<64xf32>, %arg2: memref<64x64xf32>) attributes {Kernel, forward_kernel_1} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    affine.for %arg3 = 0 to 64 step 16 {
      %0 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<64x64xf32> -> memref<16x64xf32>  {Id = "0", KernelName = "forward_kernel_1"}
      %1 = ADORA.BlockLoad %arg1 [0] : memref<64xf32> -> memref<64xf32>  {Id = "1", KernelName = "forward_kernel_1"}
      %2 = ADORA.LocalMemAlloc memref<16x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
      ADORA.kernel {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 64 step 4 {
            %3 = affine.load %0[%arg4, %arg5] : memref<16x64xf32>
            %4 = affine.load %1[%arg5] : memref<64xf32>
            %5 = arith.addf %3, %4 : f32
            affine.store %5, %2[%arg4, %arg5] : memref<16x64xf32>
            %6 = affine.apply #map(%arg5)
            %7 = affine.load %0[%arg4, %6] : memref<16x64xf32>
            %8 = affine.load %1[%6] : memref<64xf32>
            %9 = arith.addf %7, %8 : f32
            affine.store %9, %2[%arg4, %6] : memref<16x64xf32>
            %10 = affine.apply #map1(%arg5)
            %11 = affine.load %0[%arg4, %10] : memref<16x64xf32>
            %12 = affine.load %1[%10] : memref<64xf32>
            %13 = arith.addf %11, %12 : f32
            affine.store %13, %2[%arg4, %10] : memref<16x64xf32>
            %14 = affine.apply #map2(%arg5)
            %15 = affine.load %0[%arg4, %14] : memref<16x64xf32>
            %16 = affine.load %1[%14] : memref<64xf32>
            %17 = arith.addf %15, %16 : f32
            affine.store %17, %2[%arg4, %14] : memref<16x64xf32>
          }
        }
        ADORA.terminator
      } {KernelName = "forward_kernel_1"}
      ADORA.BlockStore %2, %arg2 [%arg3, 0] : memref<16x64xf32> -> memref<64x64xf32>  {Id = "2", KernelName = "forward_kernel_1"}
    }
    return
  }
}
