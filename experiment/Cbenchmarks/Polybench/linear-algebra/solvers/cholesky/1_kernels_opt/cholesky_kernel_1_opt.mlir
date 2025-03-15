#map = affine_map<(d0) -> (d0)>
module {
  func.func @cholesky_kernel_1(%arg0: memref<2000x2000xf32>, %arg1: index) attributes {Kernel, cholesky_kernel_1} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = ADORA.BlockLoad %arg0 [%arg1, %arg1] : memref<2000x2000xf32> -> memref<1x2xf32>  {Id = "0", KernelName = "cholesky_kernel_1"}
    %1 = ADORA.BlockLoad %arg0 [%arg1, 0] : memref<2000x2000xf32> -> memref<1x2000xf32>  {Id = "1", KernelName = "cholesky_kernel_1"}
    %2 = ADORA.LocalMemAlloc memref<1x2xf32>  {Id = "2", KernelName = "cholesky_kernel_1"}
    ADORA.kernel {
      %3 = affine.load %0[0, 0] : memref<1x2xf32>
      %4 = affine.for %arg2 = 0 to #map(%arg1) iter_args(%arg3 = %3) -> (f32) {
        %5 = affine.load %1[0, %arg2] : memref<1x2000xf32>
        %6 = arith.mulf %5, %5 : f32
        // %7 = arith.subf %arg3, %6 : f32
        %7 = arith.addf %arg3, %6 : f32
        affine.yield %7 : f32
      }
      affine.store %4, %2[0, 0] : memref<1x2xf32>
      ADORA.terminator
    } {KernelName = "cholesky_kernel_1"}
    ADORA.BlockStore %2, %arg0 [%arg1, %arg1] : memref<1x2xf32> -> memref<2000x2000xf32>  {Id = "2", KernelName = "cholesky_kernel_1"}
    return
  }
}