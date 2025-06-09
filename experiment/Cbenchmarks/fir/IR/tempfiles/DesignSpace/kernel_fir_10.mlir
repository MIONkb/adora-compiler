module {
  func.func @fir(%arg0: memref<100xi32>, %arg1: memref<100xi32>) -> memref<i32> {
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    %0 = ADORA.BlockLoad %arg1 [0] : memref<100xi32> -> memref<100xi32>  {Id = "0", KernelName = "kernel_fir"}
    %1 = ADORA.BlockLoad %arg0 [0] : memref<100xi32> -> memref<100xi32>  {Id = "1", KernelName = "kernel_fir"}
    %2 = ADORA.LocalMemAlloc memref<2xi32>  {Id = "2", KernelName = "kernel_fir"}
    ADORA.kernel {
      %3 = affine.for %arg2 = 0 to 100 step 10 iter_args(%arg3 = %c0_i32) -> (i32) {
        %4 = affine.load %0[%arg2] : memref<100xi32>
        %5 = affine.load %1[-%arg2 + 99] : memref<100xi32>
        %6 = arith.muli %4, %5 : i32
        %7 = arith.addi %arg3, %6 : i32
        %8 = affine.load %0[%arg2 + 1] : memref<100xi32>
        %9 = affine.load %1[-%arg2 + 98] : memref<100xi32>
        %10 = arith.muli %8, %9 : i32
        %11 = arith.addi %7, %10 : i32
        %12 = affine.load %0[%arg2 + 2] : memref<100xi32>
        %13 = affine.load %1[-%arg2 + 97] : memref<100xi32>
        %14 = arith.muli %12, %13 : i32
        %15 = arith.addi %11, %14 : i32
        %16 = affine.load %0[%arg2 + 3] : memref<100xi32>
        %17 = affine.load %1[-%arg2 + 96] : memref<100xi32>
        %18 = arith.muli %16, %17 : i32
        %19 = arith.addi %15, %18 : i32
        %20 = affine.load %0[%arg2 + 4] : memref<100xi32>
        %21 = affine.load %1[-%arg2 + 95] : memref<100xi32>
        %22 = arith.muli %20, %21 : i32
        %23 = arith.addi %19, %22 : i32
        %24 = affine.load %0[%arg2 + 5] : memref<100xi32>
        %25 = affine.load %1[-%arg2 + 94] : memref<100xi32>
        %26 = arith.muli %24, %25 : i32
        %27 = arith.addi %23, %26 : i32
        %28 = affine.load %0[%arg2 + 6] : memref<100xi32>
        %29 = affine.load %1[-%arg2 + 93] : memref<100xi32>
        %30 = arith.muli %28, %29 : i32
        %31 = arith.addi %27, %30 : i32
        %32 = affine.load %0[%arg2 + 7] : memref<100xi32>
        %33 = affine.load %1[-%arg2 + 92] : memref<100xi32>
        %34 = arith.muli %32, %33 : i32
        %35 = arith.addi %31, %34 : i32
        %36 = affine.load %0[%arg2 + 8] : memref<100xi32>
        %37 = affine.load %1[-%arg2 + 91] : memref<100xi32>
        %38 = arith.muli %36, %37 : i32
        %39 = arith.addi %35, %38 : i32
        %40 = affine.load %0[%arg2 + 9] : memref<100xi32>
        %41 = affine.load %1[-%arg2 + 90] : memref<100xi32>
        %42 = arith.muli %40, %41 : i32
        %43 = arith.addi %39, %42 : i32
        affine.yield %43 : i32
      }
      affine.store %3, %2[0] : memref<2xi32>
      ADORA.terminator
    } {KernelName = "kernel_fir"}
    ADORA.BlockStore %2, %alloca [] : memref<2xi32> -> memref<i32>  {Id = "2", KernelName = "kernel_fir"}
    return %alloca : memref<i32>
  }
}
