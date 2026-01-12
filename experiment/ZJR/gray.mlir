module attributes {} {
  func.func @gray(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c29_i32 = arith.constant 29 : i32
    %c150_i32 = arith.constant 150 : i32
    %c77_i32 = arith.constant 77 : i32
    %c24_i32 = arith.constant 24 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c255_i32 = arith.constant 255 : i32
    affine.for %arg2 = 0 to 921600 {
      %0 = affine.load %arg0[%arg2] : memref<?xi32>
      %1 = arith.shrsi %0, %c24_i32 : i32
      %2 = arith.andi %1, %c255_i32 : i32
      %3 = arith.shli %2, %c24_i32 : i32
      %4 = arith.andi %0, %c255_i32 : i32
      %5 = arith.muli %4, %c77_i32 : i32
      %6 = arith.shrsi %0, %c8_i32 : i32
      %7 = arith.andi %6, %c255_i32 : i32
      %8 = arith.muli %7, %c150_i32 : i32
      %9 = arith.addi %5, %8 : i32
      %10 = arith.shrsi %0, %c16_i32 : i32
      %11 = arith.andi %10, %c255_i32 : i32
      %12 = arith.muli %11, %c29_i32 : i32
      %13 = arith.addi %9, %12 : i32
      %14 = arith.shrsi %13, %c8_i32 : i32
      %15 = arith.shli %14, %c16_i32 : i32
      %16 = arith.ori %3, %15 : i32
      %17 = arith.shli %14, %c8_i32 : i32
      %18 = arith.ori %16, %17 : i32
      %19 = arith.ori %18, %14 : i32
      affine.store %19, %arg1[%arg2] : memref<?xi32>
    }
    return
  }
}
