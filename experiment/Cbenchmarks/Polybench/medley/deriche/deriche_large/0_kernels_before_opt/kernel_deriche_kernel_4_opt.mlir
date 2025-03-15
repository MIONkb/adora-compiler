#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1080)>
module {
  func.func @kernel_deriche_kernel_4(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<f32>, %arg4: memref<4096x2160xf32>, %arg5: memref<4096x2160xf32>) attributes {Kernel, kernel_deriche_kernel_4} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -0.183681786 : f32
    %cst_1 = arith.constant 0.114441216 : f32
    %cst_2 = arith.constant 0.840896427 : f32
    %cst_3 = arith.constant -0.606530666 : f32
    affine.for %arg6 = 0 to 4096 {
      affine.store %cst, %arg0[] : memref<f32>
      affine.store %cst, %arg1[] : memref<f32>
      affine.store %cst, %arg2[] : memref<f32>
      affine.store %cst, %arg3[] : memref<f32>
      affine.for %arg7 = 0 to 2160 step 1080 {
        %7 = ADORA.BlockLoad %arg0 [] : memref<f32> -> memref<2xf32>  {Id = "0", KernelName = ""}
        %2 = ADORA.BlockLoad %arg2 [] : memref<f32> -> memref<2xf32>  {Id = "1", KernelName = ""}
        // %2 = ADORA.BlockLoad %2 [0] : memref<2xf32> -> memref<2xf32>  {Id = "2", KernelName = ""}
        %5 = ADORA.BlockLoad %arg1 [] : memref<f32> -> memref<2xf32>  {Id = "3", KernelName = ""}
        %6 = ADORA.BlockLoad %arg3 [] : memref<f32> -> memref<2xf32>  {Id = "4", KernelName = ""}
        // %5 = ADORA.BlockLoad %5 [0] : memref<2xf32> -> memref<2xf32>  {Id = "5", KernelName = ""}
        // %6 = ADORA.BlockLoad %6 [0] : memref<2xf32> -> memref<2xf32>  {Id = "6", KernelName = ""}
        // %7 = ADORA.BlockLoad %7 [0] : memref<2xf32> -> memref<2xf32>  {Id = "7", KernelName = ""}
        %8 = ADORA.BlockLoad %arg5 [-%arg6 + 4095, %arg7] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "8", KernelName = ""}
        %9 = ADORA.BlockLoad %arg4 [-%arg6 + 4095, %arg7] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "9", KernelName = ""}
        %10 = ADORA.LocalMemAlloc memref<1x1080xf32>  {Id = "10", KernelName = ""}
        ADORA.kernel {
          affine.for %arg8 = #map(%arg7) to #map1(%arg7) {
            %11 = affine.load %7[0] : memref<2xf32>
            %12 = arith.mulf %11, %cst_0 : f32
            %13 = affine.load %5[0] : memref<2xf32>
            %14 = arith.mulf %13, %cst_1 : f32
            %15 = arith.addf %12, %14 : f32
            %16 = affine.load %2[0] : memref<2xf32>
            %17 = arith.mulf %16, %cst_2 : f32
            %18 = arith.addf %15, %17 : f32
            %19 = affine.load %6[0] : memref<2xf32>
            %20 = arith.mulf %19, %cst_3 : f32
            %21 = arith.addf %18, %20 : f32
            affine.store %21, %10[4095, %arg8] : memref<1x1080xf32>
            affine.store %11, %5[0] : memref<2xf32>
            %22 = affine.load %8[4095, %arg8] : memref<1x1080xf32>
            affine.store %22, %7[0] : memref<2xf32>
            affine.store %16, %6[0] : memref<2xf32>
            %23 = affine.load %9[4095, %arg8] : memref<1x1080xf32>
            affine.store %23, %2[0] : memref<2xf32>
          }
          ADORA.terminator
        }
        ADORA.BlockStore %10, %arg4 [-%arg6 + 4095, %arg7] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "10", KernelName = ""}
        // ADORA.BlockStore %7, %7 [0] : memref<2xf32> -> memref<2xf32>  {Id = "7", KernelName = ""}
        // ADORA.BlockStore %6, %6 [0] : memref<2xf32> -> memref<2xf32>  {Id = "6", KernelName = ""}
        // ADORA.BlockStore %5, %5 [0] : memref<2xf32> -> memref<2xf32>  {Id = "5", KernelName = ""}
        ADORA.BlockStore %6, %arg3 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = ""}
        ADORA.BlockStore %5, %arg1 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = ""}
        // ADORA.BlockStore %2, %2 [0] : memref<2xf32> -> memref<2xf32>  {Id = "2", KernelName = ""}
        ADORA.BlockStore %2, %arg2 [] : memref<2xf32> -> memref<f32>  {Id = "1", KernelName = ""}
        ADORA.BlockStore %7, %arg0 [] : memref<2xf32> -> memref<f32>  {Id = "0", KernelName = ""}
      }
    }
    return
  }
}