module {
  func.func @kernel_deriche_kernel_2(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>) attributes {Kernel, kernel_deriche_kernel_2} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    affine.for %arg3 = 0 to 4096 {
      affine.for %arg4 = 0 to 2160 step 1080 {
        %0 = ADORA.BlockLoad %arg0 [%arg3, %arg4] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "0", KernelName = "kernel_deriche_kernel_2"}
        %1 = ADORA.BlockLoad %arg1 [%arg3, %arg4] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "1", KernelName = "kernel_deriche_kernel_2"}
        %2 = ADORA.LocalMemAlloc memref<1x1080xf32>  {Id = "2", KernelName = "kernel_deriche_kernel_2"}
        ADORA.kernel {
          affine.for %arg5 = 0 to 1080 {
            %3 = affine.load %0[0, %arg5] : memref<1x1080xf32>
            %4 = affine.load %1[0, %arg5] : memref<1x1080xf32>
            %5 = arith.addf %3, %4 : f32
            affine.store %5, %2[0, %arg5] : memref<1x1080xf32>
          }
          ADORA.terminator
        } {KernelName = "kernel_deriche_kernel_2"}
        ADORA.BlockStore %2, %arg2 [%arg3, %arg4] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "2", KernelName = "kernel_deriche_kernel_2"}
      }
    }
    return
  }
}

