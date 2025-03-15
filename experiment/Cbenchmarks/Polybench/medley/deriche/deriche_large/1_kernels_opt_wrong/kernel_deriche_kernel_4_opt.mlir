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
      affine.store %cst, %arg3[] : memref<f32>
      affine.store %cst, %arg2[] : memref<f32>
      affine.store %cst, %arg1[] : memref<f32>
      affine.store %cst, %arg0[] : memref<f32>
      affine.for %arg7 = 0 to 2160 step 1080 {
        %0 = ADORA.BlockLoad %arg0 [] : memref<f32> -> memref<2xf32>  {Id = "0", KernelName = ""}
        %1 = ADORA.BlockLoad %arg2 [] : memref<f32> -> memref<2xf32>  {Id = "1", KernelName = ""}
        %2 = ADORA.BlockLoad %arg1 [] : memref<f32> -> memref<2xf32>  {Id = "3", KernelName = ""}
        %3 = ADORA.BlockLoad %arg3 [] : memref<f32> -> memref<2xf32>  {Id = "4", KernelName = ""}
        %4 = ADORA.BlockLoad %arg5 [-%arg6 + 4095, %arg7] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "8", KernelName = ""}
        %5 = ADORA.BlockLoad %arg4 [-%arg6 + 4095, %arg7] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "9", KernelName = ""}
        %6 = ADORA.LocalMemAlloc memref<1x1080xf32>  {Id = "10", KernelName = ""}
        ADORA.kernel {
          %7 = affine.load %0[0] : memref<2xf32>
          %8 = affine.load %2[0] : memref<2xf32>
          %9 = affine.load %1[0] : memref<2xf32>
          %10 = affine.load %3[0] : memref<2xf32>
          %11:4 = affine.for %arg8 = #map(%arg7) to #map1(%arg7) iter_args(%arg9 = %7, %arg10 = %8, %arg11 = %9, %arg12 = %10) -> (f32, f32, f32, f32) {
            %12 = arith.mulf %arg9, %cst_0 : f32
            %13 = arith.mulf %arg10, %cst_1 : f32
            %14 = arith.addf %12, %13 : f32
            %15 = arith.mulf %arg11, %cst_2 : f32
            %16 = arith.addf %14, %15 : f32
            %17 = arith.mulf %arg12, %cst_3 : f32
            %18 = arith.addf %16, %17 : f32
            affine.store %18, %6[4095, %arg8] : memref<1x1080xf32>
            %19 = affine.load %4[4095, %arg8] : memref<1x1080xf32>
            %20 = affine.load %5[4095, %arg8] : memref<1x1080xf32>
            affine.yield %19, %arg9, %20, %arg11 : f32, f32, f32, f32
          }
          affine.store %11#3, %3[0] : memref<2xf32>
          affine.store %11#2, %1[0] : memref<2xf32>
          affine.store %11#1, %2[0] : memref<2xf32>
          affine.store %11#0, %0[0] : memref<2xf32>
          ADORA.terminator
        }
        ADORA.BlockStore %6, %arg4 [-%arg6 + 4095, %arg7] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "10", KernelName = ""}
        ADORA.BlockStore %3, %arg3 [] : memref<2xf32> -> memref<f32>  {Id = "4", KernelName = ""}
        ADORA.BlockStore %2, %arg1 [] : memref<2xf32> -> memref<f32>  {Id = "3", KernelName = ""}
        ADORA.BlockStore %1, %arg2 [] : memref<2xf32> -> memref<f32>  {Id = "1", KernelName = ""}
        ADORA.BlockStore %0, %arg0 [] : memref<2xf32> -> memref<f32>  {Id = "0", KernelName = ""}
      }
    }

    return
  }
}