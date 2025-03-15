module attributes {torch.debug_module_name = "ReLU"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @forward(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.getelementptr %2[8192] : (!llvm.ptr) -> !llvm.ptr, f32
    %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.add %4, %0  : i64
    %6 = llvm.call @malloc(%5) : (i64) -> !llvm.ptr
    %7 = llvm.ptrtoint %6 : !llvm.ptr to i64
    %8 = llvm.sub %0, %1  : i64
    %9 = llvm.add %7, %8  : i64
    %10 = llvm.urem %9, %0  : i64
    %11 = llvm.sub %9, %10  : i64
    %12 = llvm.inttoptr %11 : i64 to !llvm.ptr
    llvm.call @forward_kernel_0(%arg0, %12) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return %6 : !llvm.ptr
  }
  llvm.func @forward_kernel_0(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}

