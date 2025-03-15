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
      affine.store %cst, %arg3[] : memref<f32>
      affine.store %cst, %arg2[] : memref<f32>
      affine.store %cst, %arg1[] : memref<f32>
      affine.store %cst, %arg0[] : memref<f32>
      affine.for %arg7 = 0 to 2160 step 1080 {
        %0 = ADORA.BlockLoad %arg2 [] : memref<f32> -> memref<2xf32>  {Id = "0", KernelName = ""}
        %1 = ADORA.BlockLoad %arg0 [] : memref<f32> -> memref<2xf32>  {Id = "1", KernelName = ""}
        %2 = ADORA.BlockLoad %arg3 [] : memref<f32> -> memref<2xf32>  {Id = "3", KernelName = ""}
        %3 = ADORA.BlockLoad %arg1 [] : memref<f32> -> memref<2xf32>  {Id = "4", KernelName = ""}
        %4 = ADORA.BlockLoad %arg5 [%arg6, -%arg7 + 1080] : memref<4096x2160xf32> -> memref<1x1080xf32>  {Id = "8", KernelName = ""}
        %5 = ADORA.LocalMemAlloc memref<1x1080xf32>  {Id = "9", KernelName = ""}
        %a0 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "5", KernelName = ""}
        %a1 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "6", KernelName = ""}
        %a2 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "7", KernelName = ""}
        %a3 = ADORA.LocalMemAlloc memref<2xf32>  {Id = "8", KernelName = ""}        
        ADORA.kernel {
          %6 = affine.load %0[0] : memref<2xf32>
          %7 = affine.load %2[0] : memref<2xf32>
          %8 = affine.load %1[0] : memref<2xf32>
          %9 = affine.load %3[0] : memref<2xf32>
          %10:4 = affine.for %arg8 = #map(%arg7) to #map1(%arg7) iter_args(%arg9 = %6, %arg10 = %7, %arg11 = %8, %arg12 = %9) -> (f32, f32, f32, f32) {
            %11 = arith.mulf %arg9, %cst_0 : f32
            %12 = arith.mulf %arg10, %cst_1 : f32
            %13 = arith.addf %11, %12 : f32
            %14 = arith.mulf %arg11, %cst_2 : f32
            %15 = arith.addf %13, %14 : f32
            %16 = arith.mulf %arg12, %cst_3 : f32
            %17 = arith.addf %15, %16 : f32
            affine.store %17, %5[0, -%arg8 + 2159] : memref<1x1080xf32>
            %18 = affine.load %4[0, -%arg8 + 2159] : memref<1x1080xf32>
            affine.yield %18, %arg9, %17, %arg11 : f32, f32, f32, f32
          }
          affine.store %10#3, %a3[0] : memref<2xf32>
          affine.store %10#2, %a1[0] : memref<2xf32>
          affine.store %10#1, %a2[0] : memref<2xf32>
          affine.store %10#0, %a0[0] : memref<2xf32>
          ADORA.terminator
        }
        ADORA.BlockStore %5, %arg4 [%arg6, -%arg7 + 1080] : memref<1x1080xf32> -> memref<4096x2160xf32>  {Id = "9", KernelName = ""}
        ADORA.BlockStore %a3, %arg1 [] : memref<2xf32> -> memref<f32>  {Id = "8", KernelName = ""}
        ADORA.BlockStore %a2, %arg3 [] : memref<2xf32> -> memref<f32>  {Id = "7", KernelName = ""}
        ADORA.BlockStore %a1, %arg0 [] : memref<2xf32> -> memref<f32>  {Id = "6", KernelName = ""}
        ADORA.BlockStore %a0, %arg2 [] : memref<2xf32> -> memref<f32>  {Id = "5", KernelName = ""}
      }
    }

    return
  }
}