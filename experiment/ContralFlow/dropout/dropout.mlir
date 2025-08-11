module  {
  func.func @dropout(%arg0: memref<1024xi32>, %arg1: memref<128xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c8 = arith.constant 8 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0:2 = affine.for %arg2 = 0 to 1024 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32) -> (i32, i32) {
      %1 = arith.index_cast %arg2 : index to i32
      %2 = arith.remsi %arg2, %c8 : index
      %3 = arith.cmpi slt, %2, %c0 : index
      %4 = arith.addi %2, %c8 : index
      %5 = arith.select %3, %4, %2 : index
      %6 = arith.cmpi eq, %5, %c0 : index
      %7 = scf.if %6 -> (i32) {
        %13 = arith.shrui %1, %c3_i32 : i32
        %14 = arith.index_cast %13 : i32 to index
        %15 = memref.load %arg1[%14] : memref<128xi32>
        scf.yield %15 : i32
      } else {
        scf.yield %arg4 : i32
      }
      %8 = arith.andi %7, %c1_i32 : i32
      %9 = arith.cmpi ne, %8, %c0_i32 : i32
      %10 = scf.if %9 -> (i32) {
        %13 = affine.load %arg0[%arg2] : memref<1024xi32>
        %14 = arith.muli %13, %c2_i32 : i32
        scf.yield %14 : i32
      } else {
        scf.yield %c0_i32 : i32
      }
      %11 = arith.addi %arg3, %10 : i32
      %12 = arith.shrsi %7, %c1_i32 : i32
      affine.yield %11, %12 : i32, i32
    }
    return %0#0 : i32
  }
}