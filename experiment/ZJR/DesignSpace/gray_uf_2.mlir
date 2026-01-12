#map = affine_map<(d0) -> (d0 + 2048)>
module {
  func.func @gray(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c29_i32 = arith.constant 29 : i32
    %c150_i32 = arith.constant 150 : i32
    %c77_i32 = arith.constant 77 : i32
    %c24_i32 = arith.constant 24 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c255_i32 = arith.constant 255 : i32
    affine.for %arg2 = 0 to 921600 step 4096 {
      %0 = ADORA.BlockLoad %arg0 [%arg2] : memref<?xi32> -> memref<2048xi32>  {Id = "0", KernelName = "gray"}
      %1 = ADORA.LocalMemAlloc memref<2048xi32>  {Id = "1", KernelName = "gray"}
      %2 = affine.apply #map(%arg2)
      %3 = ADORA.BlockLoad %arg0 [%2] : memref<?xi32> -> memref<2048xi32>  {Id = "0", KernelName = "gray"}
      %4 = ADORA.LocalMemAlloc memref<2048xi32>  {Id = "1", KernelName = "gray"}
      ADORA.kernel {
        affine.for %arg3 = 0 to 2048 {
          %6 = affine.load %0[%arg3] : memref<2048xi32>
          %7 = arith.shrsi %6, %c24_i32 : i32
          %8 = arith.andi %7, %c255_i32 : i32
          %9 = arith.shli %8, %c24_i32 : i32
          %10 = arith.andi %6, %c255_i32 : i32
          %11 = arith.muli %10, %c77_i32 : i32
          %12 = arith.shrsi %6, %c8_i32 : i32
          %13 = arith.andi %12, %c255_i32 : i32
          %14 = arith.muli %13, %c150_i32 : i32
          %15 = arith.addi %11, %14 : i32
          %16 = arith.shrsi %6, %c16_i32 : i32
          %17 = arith.andi %16, %c255_i32 : i32
          %18 = arith.muli %17, %c29_i32 : i32
          %19 = arith.addi %15, %18 : i32
          %20 = arith.shrsi %19, %c8_i32 : i32
          %21 = arith.shli %20, %c16_i32 : i32
          %22 = arith.ori %9, %21 : i32
          %23 = arith.shli %20, %c8_i32 : i32
          %24 = arith.ori %22, %23 : i32
          %25 = arith.ori %24, %20 : i32
          affine.store %25, %1[%arg3] : memref<2048xi32>
          %26 = affine.apply #map(%arg2)
          %27 = affine.load %3[%arg3] : memref<2048xi32>
          %28 = arith.shrsi %27, %c24_i32 : i32
          %29 = arith.andi %28, %c255_i32 : i32
          %30 = arith.shli %29, %c24_i32 : i32
          %31 = arith.andi %27, %c255_i32 : i32
          %32 = arith.muli %31, %c77_i32 : i32
          %33 = arith.shrsi %27, %c8_i32 : i32
          %34 = arith.andi %33, %c255_i32 : i32
          %35 = arith.muli %34, %c150_i32 : i32
          %36 = arith.addi %32, %35 : i32
          %37 = arith.shrsi %27, %c16_i32 : i32
          %38 = arith.andi %37, %c255_i32 : i32
          %39 = arith.muli %38, %c29_i32 : i32
          %40 = arith.addi %36, %39 : i32
          %41 = arith.shrsi %40, %c8_i32 : i32
          %42 = arith.shli %41, %c16_i32 : i32
          %43 = arith.ori %30, %42 : i32
          %44 = arith.shli %41, %c8_i32 : i32
          %45 = arith.ori %43, %44 : i32
          %46 = arith.ori %45, %41 : i32
          affine.store %46, %4[%arg3] : memref<2048xi32>
        }
        ADORA.terminator
      } {KernelName = "gray"}
      ADORA.BlockStore %1, %arg1 [%arg2] : memref<2048xi32> -> memref<?xi32>  {Id = "1", KernelName = "gray"}
      %5 = affine.apply #map(%arg2)
      ADORA.BlockStore %4, %arg1 [%5] : memref<2048xi32> -> memref<?xi32>  {Id = "1", KernelName = "gray"}
    }
    return
  }
}
