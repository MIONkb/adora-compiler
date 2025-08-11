module {
  func.func @dropout(%arg0: memref<1024xi32>, %arg1: memref<128xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c7 = arith.constant 7 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c8 = arith.constant 8 : index
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %2:2 = dataflow.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32) -> (i32, i32) {
          %3:2 = dataflow.execution : i32, i32 {
            %4 = arith.index_cast %arg2 : index to i32
            %5 = arith.andi %arg2, %c7 : index
            %6 = arith.cmpi slt, %5, %c0 : index
            %7 = arith.addi %5, %c8 : index
            %8 = arith.select %6, %7, %5 : index
            %9 = arith.cmpi eq, %8, %c0 : index
            %10 = arith.shrui %4, %c3_i32 : i32
            %11 = arith.index_cast %10 : i32 to index
            %12 = dataflow.addr %arg1[%11] {memShape = [128]} : memref<128xi32>[index] -> i32
            %13 = dataflow.load %12 : i32 -> i32
            %14 = dataflow.select %9, %13, %arg4 : i32
            %15 = arith.andi %14, %c1_i32 : i32
            %16 = arith.cmpi ne, %15, %c0_i32 : i32
            %17 = dataflow.addr %arg0[%arg2] {memShape = [1024]} : memref<1024xi32>[index] -> i32
            %18 = dataflow.load %17 : i32 -> i32
            %19 = arith.muli %18, %c2_i32 : i32
            %20 = dataflow.select %16, %19, %c0_i32 : i32
            %21 = arith.addi %arg3, %20 : i32
            %22 = arith.shrsi %14, %c1_i32 : i32
            %23 = arith.addi %arg2, %c1 {Exe = "Loop"} : index
            %24 = arith.cmpi eq, %23, %c1024 {Exe = "Loop"} : index
            dataflow.state %24, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
            dataflow.yield {execution_block = 1 : i32} %21, %22 : i32, i32
          }
          dataflow.yield %3#0, %3#1 : i32, i32
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        dataflow.yield %2#0 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
}