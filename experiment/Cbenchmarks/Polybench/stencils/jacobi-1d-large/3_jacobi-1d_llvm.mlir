module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @jacobi_1d(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.mlir.constant(500 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%2 : i64)
  ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb2
    %4 = llvm.icmp "slt" %3, %0 : i64
    llvm.cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @jacobi_1d_kernel_0(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @jacobi_1d_kernel_1(%arg1, %arg0) : (!llvm.ptr, !llvm.ptr) -> ()
    %5 = llvm.add %3, %1  : i64
    llvm.br ^bb1(%5 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @jacobi_1d_kernel_0(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @jacobi_1d_kernel_1(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}

