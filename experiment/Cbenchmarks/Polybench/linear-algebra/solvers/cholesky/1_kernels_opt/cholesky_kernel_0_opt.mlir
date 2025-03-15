func.func @cholesky_kernel_0(%arg0: memref<2000x2000xf32>, %arg1: index, %arg2: index) attributes {Kernel, cholesky_kernel_0} {
  cf.br ^bb1
^bb1:  // pred: ^bb0
    %a = ADORA.BlockLoad %arg0 [%arg1, 0] : memref<2000x2000xf32> -> memref<1x2000xf32>  {Id = "0", KernelName = "cholesky_kernel_0"}
    %b = ADORA.BlockLoad %arg0 [%arg2, 0] : memref<2000x2000xf32> -> memref<1x2000xf32>  {Id = "1", KernelName = "cholesky_kernel_0"}
    %c = ADORA.BlockLoad %arg0 [%arg1, %arg2] : memref<2000x2000xf32> -> memref<1x2xf32>  {Id = "2", KernelName = "cholesky_kernel_0"}
    %d = ADORA.LocalMemAlloc memref<1x2xf32>  {Id = "3", KernelName = "cholesky_kernel_1"}
    ADORA.kernel{
    affine.for %arg3 = 0 to affine_map<(d0) -> (d0)>(%arg2) {
      %3 = affine.load %a[0, %arg3] : memref<1x2000xf32>
      %4 = affine.load %b[0, %arg3] : memref<1x2000xf32>
      %5 = arith.mulf %3, %4 : f32
      %6 = affine.load %c[0, 0] : memref<1x2xf32>
      // %7 = arith.subf %6, %5 : f32
      %7 = arith.addf %6, %5 : f32
      affine.store %7, %d[0, 0] : memref<1x2xf32>
    }
    ADORA.terminator
    } {KernelName = "cholesky_kernel_0"}
    ADORA.BlockStore %d, %arg0 [%arg1, %arg2] : memref<1x2xf32> -> memref<2000x2000xf32>  {Id = "2", KernelName = "cholesky_kernel_0"}
  return
}