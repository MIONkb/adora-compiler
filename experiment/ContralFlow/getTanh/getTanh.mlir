module  {
  func.func @getTanh(%arg0: memref<100xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3_i32 = arith.constant 3 : i32
    %c19_i32 = arith.constant 19 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg1 = 0 to 100 iter_args(%arg2 = %c0_i32) -> (i32) {
      %1 = affine.load %arg0[%arg1] : memref<100xi32>
      %2 = arith.cmpi slt, %1, %c1_i32 : i32
      %3 = scf.if %2 -> (i32) {
        %5 = arith.muli %1, %1 : i32
        %6 = arith.addi %5, %c19_i32 : i32
        %7 = arith.muli %6, %1 : i32
        %8 = arith.muli %7, %1 : i32
        %9 = arith.addi %8, %c3_i32 : i32
        %10 = arith.muli %9, %1 : i32
        scf.yield %10 : i32
      } else {
        scf.yield %c1_i32 : i32
      }

      ////
      //   %5 = arith.muli %1, %1 : i32
      //   %6 = arith.addi %5, %c19_i32 : i32
      //   %7 = arith.muli %6, %1 : i32
      //   %8 = arith.muli %7, %1 : i32
      //   %9 = arith.addi %8, %c3_i32 : i32
      //   %10 = arith.muli %9, %1 : i32      
      // %c = compare %2, 0
      // %3 = sel %c, %10, %c1_i32
      ////

      %4 = arith.addi %arg2, %3 : i32
      affine.yield %4 : i32
    }
    return %0 : i32
  }
}