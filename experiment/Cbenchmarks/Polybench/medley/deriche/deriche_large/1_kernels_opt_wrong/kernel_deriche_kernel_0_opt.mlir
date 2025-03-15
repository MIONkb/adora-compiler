#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1080)>
module {
  func.func @kernel_deriche_kernel_0(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>) attributes {Kernel, kernel_deriche_kernel_0} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -0.188681662 : f32
    %cst_1 = arith.constant 0.110209078 : f32
    %cst_2 = arith.constant 0.840896427 : f32
    %cst_3 = arith.constant -0.606530666 : f32
    affine.for %arg5 = 0 to 4096 {
      affine.for %arg6 = 0 to 2160 step 1080 {
        %0 = ADORA.BlockLoad %arg2 [] : memref<f32> -> memref<2xf32>  {Id = "0", KernelName = ""}
        %1 = ADORA.BlockLoad %arg1 [] : memref<f32> -> memref<2xf32>  {Id = "1", KernelName = ""}
        %2 = ADORA.BlockLoad %arg0 [] : memref<f32> -> memref<2xf32>  {Id = "3", KernelName = ""}
        %3 = ADORA.BlockLoad %arg3 [%arg5, %arg6] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "6", KernelName = ""}
        %4 = ADORA.LocalMemAlloc memref<1x1080xf32>  {Id = "7", KernelName = ""}

        %a0 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "8", KernelName = ""}
        %a2 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "9", KernelName = ""}
        %a1 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "10", KernelName = ""}
        ADORA.kernel {
          %5 = affine.load %0[0] : memref<2xf32>
          %6 = affine.load %2[0] : memref<2xf32>
          %7 = affine.load %1[0] : memref<2xf32>
          %8:3 = affine.for %arg7 = 0 to 1080 iter_args(%arg8 = %5, %arg9 = %6, %arg10 = %7) -> (f32, f32, f32) {
            %9 = affine.load %3[0, %arg7] : memref<1x1080xf32>
            %10 = arith.mulf %9, %cst_0 : f32
            %11 = arith.mulf %arg8, %cst_1 : f32
            %12 = arith.addf %10, %11 : f32
            %13 = arith.mulf %arg9, %cst_2 : f32
            %14 = arith.addf %12, %13 : f32
            %15 = arith.mulf %arg10, %cst_3 : f32
            %16 = arith.addf %14, %15 : f32
            affine.store %16, %4[0, %arg7] : memref<1x1080xf32>
            affine.yield %9, %16, %arg9 : f32, f32, f32
          }
          affine.store %8#2, %a1[0] : memref<2xf32>
          affine.store %8#1, %a2[0] : memref<2xf32>
          affine.store %8#0, %a0[0] : memref<2xf32>
          ADORA.terminator
        }
        ADORA.BlockStore %4, %arg4 [%arg5, %arg6] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "7", KernelName = ""}
        ADORA.BlockStore %a2, %arg0 [] : memref<2xf32> -> memref<f32>  {Id = "10", KernelName = ""}
        ADORA.BlockStore %a1, %arg1 [] : memref<2xf32> -> memref<f32>  {Id = "9", KernelName = ""}
        ADORA.BlockStore %a0, %arg2 [] : memref<2xf32> -> memref<f32>  {Id = "8", KernelName = ""}
      }
    }
    affine.store %cst, %arg2[] : memref<f32>
    affine.store %cst, %arg1[] : memref<f32>
    affine.store %cst, %arg0[] : memref<f32>
    return
  }
}