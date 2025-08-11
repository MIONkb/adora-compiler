module {
  func.func @getTanh(%arg0: memref<100xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c19_i32 = arith.constant 19 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %2 = dataflow.for %arg1 = %c0 to %c100 step %c1 iter_args(%arg2 = %c0_i32) -> (i32) {
          %3 = dataflow.execution : i32 {
            %4 = dataflow.addr %arg0[%arg1] {memShape = [100]} : memref<100xi32>[index] -> i32
            %5 = dataflow.load %4 : i32 -> i32
            %6 = arith.cmpi slt, %5, %c1_i32 : i32
            %7 = arith.muli %5, %5 : i32
            %8 = arith.addi %7, %c19_i32 : i32
            %9 = arith.muli %8, %5 : i32
            %10 = arith.muli %9, %5 : i32
            %11 = arith.addi %10, %c3_i32 : i32
            %12 = arith.muli %11, %5 : i32
            %13 = dataflow.select %6, %12, %c1_i32 : i32
            %14 = arith.addi %arg2, %13 : i32
            %15 = arith.addi %arg1, %c1 {Exe = "Loop"} : index
            %16 = arith.cmpi eq, %15, %c100 {Exe = "Loop"} : index
            dataflow.state %16, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
            dataflow.yield {execution_block = 1 : i32} %14 : i32
          }
          dataflow.yield %3 : i32
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        dataflow.yield %2 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
}