#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1080)>
module {
  func.func @kernel_deriche_kernel_1(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<f32>, %arg4: memref<4096x2160xf32>, %arg5: memref<4096x2160xf32>) attributes {Kernel, kernel_deriche_kernel_1} {
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
        %7 = ADORA.BlockLoad %arg2 [] : memref<f32> -> memref<2xf32>  {Id = "0", KernelName = ""}
        %2 = ADORA.BlockLoad %arg0 [] : memref<f32> -> memref<2xf32>  {Id = "1", KernelName = ""}
        // %2 = ADORA.BlockLoad %1 [0] : memref<2xf32> -> memref<2xf32>  {Id = "2", KernelName = ""}
        %5 = ADORA.BlockLoad %arg3 [] : memref<f32> -> memref<2xf32>  {Id = "3", KernelName = ""}
        %6 = ADORA.BlockLoad %arg1 [] : memref<f32> -> memref<2xf32>  {Id = "4", KernelName = ""}
        // %5 = ADORA.BlockLoad %3 [0] : memref<2xf32> -> memref<2xf32>  {Id = "5", KernelName = ""}
        // %6 = ADORA.BlockLoad %4 [0] : memref<2xf32> -> memref<2xf32>  {Id = "6", KernelName = ""}
        // %7 = ADORA.BlockLoad %0 [0] : memref<2xf32> -> memref<2xf32>  {Id = "7", KernelName = ""}
        %8 = ADORA.BlockLoad %arg5 [%arg6, -%arg7 + 1080] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "8", KernelName = ""}
        %9 = ADORA.LocalMemAlloc memref<1x1080xf32>  {Id = "9", KernelName = ""}
        ADORA.kernel {
          affine.for %arg8 = #map(%arg7) to #map1(%arg7) {
            %10 = affine.load %7[0] : memref<2xf32>
            %11 = arith.mulf %10, %cst_0 : f32
            %12 = affine.load %5[0] : memref<2xf32>
            %13 = arith.mulf %12, %cst_1 : f32
            %14 = arith.addf %11, %13 : f32
            %15 = affine.load %2[0] : memref<2xf32>
            %16 = arith.mulf %15, %cst_2 : f32
            %17 = arith.addf %14, %16 : f32
            %18 = affine.load %6[0] : memref<2xf32>
            %19 = arith.mulf %18, %cst_3 : f32
            %20 = arith.addf %17, %19 : f32
            affine.store %20, %9[0, -%arg8 + 2159] : memref<1x1080xf32>
            affine.store %10, %5[0] : memref<2xf32>
            %21 = affine.load %8[0, -%arg8 + 2159] : memref<1x1080xf32>
            affine.store %21, %7[0] : memref<2xf32>
            affine.store %15, %6[0] : memref<2xf32>
            affine.store %20, %2[0] : memref<2xf32>
          }
          ADORA.terminator
        }
        ADORA.BlockStore %9, %arg4 [%arg6, -%arg7 + 1080] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "9", KernelName = ""}
        // ADORA.BlockStore %7, %0 [0] : memref<2xf32> -> memref<2xf32>  {Id = "7", KernelName = ""}
        // ADORA.BlockStore %6, %4 [0] : memref<2xf32> -> memref<2xf32>  {Id = "6", KernelName = ""}
        // ADORA.BlockStore %5, %3 [0] : memref<2xf32> -> memref<2xf32>  {Id = "5", KernelName = ""}
        ADORA.BlockStore %6, %arg1 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = ""}
        ADORA.BlockStore %5, %arg3 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = ""}
        // ADORA.BlockStore %2, %1 [0] : memref<2xf32> -> memref<2xf32>  {Id = "2", KernelName = ""}
        ADORA.BlockStore %2, %arg0 [] : memref<2xf32> -> memref<f32>  {Id = "1", KernelName = ""}
        ADORA.BlockStore %7, %arg2 [] : memref<2xf32> -> memref<f32>  {Id = "0", KernelName = ""}
      }
    }
    return
  }
}