#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1080)>
module {
  func.func @kernel_deriche_kernel_3(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>) attributes {Kernel, kernel_deriche_kernel_3} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -0.188681662 : f32
    %cst_1 = arith.constant 0.110209078 : f32
    %cst_2 = arith.constant 0.840896427 : f32
    %cst_3 = arith.constant -0.606530666 : f32
    affine.for %arg5 = 0 to 4096 {
      affine.store %cst, %arg0[] : memref<f32>
      affine.store %cst, %arg1[] : memref<f32>
      affine.store %cst, %arg2[] : memref<f32>
      affine.for %arg6 = 0 to 2160 step 1080 {
        %4 = ADORA.BlockLoad %arg0 [] : memref<f32> -> memref<2xf32>  {Id = "0", KernelName = ""}
        %2 = ADORA.BlockLoad %arg2 [] : memref<f32> -> memref<2xf32>  {Id = "1", KernelName = ""}
        // %2 = ADORA.BlockLoad %2 [0] : memref<2xf32> -> memref<2xf32>  {Id = "2", KernelName = ""}
        %5 = ADORA.BlockLoad %arg1 [] : memref<f32> -> memref<2xf32>  {Id = "3", KernelName = ""}
        // %4 = ADORA.BlockLoad %4 [0] : memref<2xf32> -> memref<2xf32>  {Id = "4", KernelName = ""}
        // %5 = ADORA.BlockLoad %5 [0] : memref<2xf32> -> memref<2xf32>  {Id = "5", KernelName = ""}
        %6 = ADORA.BlockLoad %arg3 [%arg5, %arg6] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "6", KernelName = ""}
        %7 = ADORA.BlockLoad %arg3 [%arg5, %arg6] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "7", KernelName = ""}
        %8 = ADORA.BlockLoad %arg4 [%arg5, %arg6] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "8", KernelName = ""}
        %9 = ADORA.LocalMemAlloc memref<1x1080xf32>  {Id = "9", KernelName = ""}
        ADORA.kernel {
          affine.for %arg7 = #map(%arg6) to #map1(%arg6) {
            %10 = affine.load %6[0, %arg7] : memref<1x1080xf32>
            %11 = arith.mulf %10, %cst_0 : f32
            %12 = affine.load %4[0] : memref<2xf32>
            %13 = arith.mulf %12, %cst_1 : f32
            %14 = arith.addf %11, %13 : f32
            %15 = affine.load %5[0] : memref<2xf32>
            %16 = arith.mulf %15, %cst_2 : f32
            %17 = arith.addf %14, %16 : f32
            %18 = affine.load %2[0] : memref<2xf32>
            %19 = arith.mulf %18, %cst_3 : f32
            %20 = arith.addf %17, %19 : f32
            affine.store %20, %9[0, %arg7] : memref<1x1080xf32>
            %21 = affine.load %7[0, %arg7] : memref<1x1080xf32>
            affine.store %21, %4[0] : memref<2xf32>
            affine.store %15, %2[0] : memref<2xf32>
            %22 = affine.load %8[0, %arg7] : memref<1x1080xf32>
            affine.store %22, %5[0] : memref<2xf32>
          }
          ADORA.terminator
        }
        ADORA.BlockStore %9, %arg4 [%arg5, %arg6] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "9", KernelName = ""}
        // ADORA.BlockStore %5, %5 [0] : memref<2xf32> -> memref<2xf32>  {Id = "5", KernelName = ""}
        // ADORA.BlockStore %4, %4 [0] : memref<2xf32> -> memref<2xf32>  {Id = "4", KernelName = ""}
        ADORA.BlockStore %5, %arg1 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = ""}
        // ADORA.BlockStore %2, %2 [0] : memref<2xf32> -> memref<2xf32>  {Id = "2", KernelName = ""}
        ADORA.BlockStore %2, %arg2 [] : memref<2xf32> -> memref<f32>  {Id = "1", KernelName = ""}
        ADORA.BlockStore %4, %arg0 [] : memref<2xf32> -> memref<f32>  {Id = "0", KernelName = ""}
      }
    }
    return
  }
}