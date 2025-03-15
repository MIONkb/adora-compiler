#map = affine_map<(d0) -> (d0 + 16)>
#map1 = affine_map<(d0) -> (d0 + 32)>
#map2 = affine_map<(d0) -> (d0 + 48)>
#map3 = affine_map<(d0) -> (d0 + 64)>
#map4 = affine_map<(d0) -> (d0 + 80)>
#map5 = affine_map<(d0) -> (d0 + 96)>
module {
  func.func @forward_kernel_2(%arg0: memref<1x64x112x112xf32>, %arg1: memref<1x64x112x112xf32>) attributes {Kernel, forward_kernel_2} {
    %c0 = arith.constant 0 : index
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        %0 = ADORA.BlockLoad %arg0 [0, %arg3, %c0, 0] : memref<1x64x112x112xf32> -> memref<1x1x16x112xf32>  {Id = "0", KernelName = "forward_kernel_2"}
        %1 = ADORA.LocalMemAlloc memref<1x1x16x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
        %2 = affine.apply #map(%c0)
        %3 = ADORA.BlockLoad %arg0 [0, %arg3, %2, 0] : memref<1x64x112x112xf32> -> memref<1x1x16x112xf32>  {Id = "0", KernelName = "forward_kernel_2"}
        %4 = ADORA.LocalMemAlloc memref<1x1x16x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
        %5 = affine.apply #map1(%c0)
        %6 = ADORA.BlockLoad %arg0 [0, %arg3, %5, 0] : memref<1x64x112x112xf32> -> memref<1x1x16x112xf32>  {Id = "0", KernelName = "forward_kernel_2"}
        %7 = ADORA.LocalMemAlloc memref<1x1x16x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
        %8 = affine.apply #map2(%c0)
        %9 = ADORA.BlockLoad %arg0 [0, %arg3, %8, 0] : memref<1x64x112x112xf32> -> memref<1x1x16x112xf32>  {Id = "0", KernelName = "forward_kernel_2"}
        %10 = ADORA.LocalMemAlloc memref<1x1x16x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
        %11 = affine.apply #map3(%c0)
        %12 = ADORA.BlockLoad %arg0 [0, %arg3, %11, 0] : memref<1x64x112x112xf32> -> memref<1x1x16x112xf32>  {Id = "0", KernelName = "forward_kernel_2"}
        %13 = ADORA.LocalMemAlloc memref<1x1x16x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
        %14 = affine.apply #map4(%c0)
        %15 = ADORA.BlockLoad %arg0 [0, %arg3, %14, 0] : memref<1x64x112x112xf32> -> memref<1x1x16x112xf32>  {Id = "0", KernelName = "forward_kernel_2"}
        %16 = ADORA.LocalMemAlloc memref<1x1x16x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
        %17 = affine.apply #map5(%c0)
        %18 = ADORA.BlockLoad %arg0 [0, %arg3, %17, 0] : memref<1x64x112x112xf32> -> memref<1x1x16x112xf32>  {Id = "0", KernelName = "forward_kernel_2"}
        %19 = ADORA.LocalMemAlloc memref<1x1x16x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
        ADORA.kernel {
          affine.for %arg4 = 0 to 16 {
            affine.for %arg5 = 0 to 112 {
              %26 = affine.load %0[0, 0, %arg4, %arg5] : memref<1x1x16x112xf32>
              %27 = arith.cmpf ugt, %26, %cst : f32
              %28 = arith.select %27, %26, %cst : f32
              affine.store %28, %1[0, 0, %arg4, %arg5] : memref<1x1x16x112xf32>
              %29 = affine.apply #map(%c0)
              %30 = affine.load %3[0, 0, %arg4, %arg5] : memref<1x1x16x112xf32>
              %31 = arith.cmpf ugt, %30, %cst : f32
              %32 = arith.select %31, %30, %cst : f32
              affine.store %32, %4[0, 0, %arg4, %arg5] : memref<1x1x16x112xf32>
              %33 = affine.apply #map1(%c0)
              %34 = affine.load %6[0, 0, %arg4, %arg5] : memref<1x1x16x112xf32>
              %35 = arith.cmpf ugt, %34, %cst : f32
              %36 = arith.select %35, %34, %cst : f32
              affine.store %36, %7[0, 0, %arg4, %arg5] : memref<1x1x16x112xf32>
              %37 = affine.apply #map2(%c0)
              %38 = affine.load %9[0, 0, %arg4, %arg5] : memref<1x1x16x112xf32>
              %39 = arith.cmpf ugt, %38, %cst : f32
              %40 = arith.select %39, %38, %cst : f32
              affine.store %40, %10[0, 0, %arg4, %arg5] : memref<1x1x16x112xf32>
              %41 = affine.apply #map3(%c0)
              %42 = affine.load %12[0, 0, %arg4, %arg5] : memref<1x1x16x112xf32>
              %43 = arith.cmpf ugt, %42, %cst : f32
              %44 = arith.select %43, %42, %cst : f32
              affine.store %44, %13[0, 0, %arg4, %arg5] : memref<1x1x16x112xf32>
              %45 = affine.apply #map4(%c0)
              %46 = affine.load %15[0, 0, %arg4, %arg5] : memref<1x1x16x112xf32>
              %47 = arith.cmpf ugt, %46, %cst : f32
              %48 = arith.select %47, %46, %cst : f32
              affine.store %48, %16[0, 0, %arg4, %arg5] : memref<1x1x16x112xf32>
              %49 = affine.apply #map5(%c0)
              %50 = affine.load %18[0, 0, %arg4, %arg5] : memref<1x1x16x112xf32>
              %51 = arith.cmpf ugt, %50, %cst : f32
              %52 = arith.select %51, %50, %cst : f32
              affine.store %52, %19[0, 0, %arg4, %arg5] : memref<1x1x16x112xf32>
            }
          }
          ADORA.terminator
        } {KernelName = "forward_kernel_2"}
        ADORA.BlockStore %1, %arg1 [0, %arg3, %c0, 0] : memref<1x1x16x112xf32> -> memref<1x64x112x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
        %20 = affine.apply #map(%c0)
        ADORA.BlockStore %4, %arg1 [0, %arg3, %20, 0] : memref<1x1x16x112xf32> -> memref<1x64x112x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
        %21 = affine.apply #map1(%c0)
        ADORA.BlockStore %7, %arg1 [0, %arg3, %21, 0] : memref<1x1x16x112xf32> -> memref<1x64x112x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
        %22 = affine.apply #map2(%c0)
        ADORA.BlockStore %10, %arg1 [0, %arg3, %22, 0] : memref<1x1x16x112xf32> -> memref<1x64x112x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
        %23 = affine.apply #map3(%c0)
        ADORA.BlockStore %13, %arg1 [0, %arg3, %23, 0] : memref<1x1x16x112xf32> -> memref<1x64x112x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
        %24 = affine.apply #map4(%c0)
        ADORA.BlockStore %16, %arg1 [0, %arg3, %24, 0] : memref<1x1x16x112xf32> -> memref<1x64x112x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
        %25 = affine.apply #map5(%c0)
        ADORA.BlockStore %19, %arg1 [0, %arg3, %25, 0] : memref<1x1x16x112xf32> -> memref<1x64x112x112xf32>  {Id = "1", KernelName = "forward_kernel_2"}
      }
    }
    return
  }
}
