module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @cholesky_kernel_0(!llvm.ptr, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @cholesky_kernel_1(!llvm.ptr, i64) attributes {sym_visibility = "private"}
  llvm.func @cholesky(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(2000 : index) : i64
    %2 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%2 : i64)
  ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb5
    %4 = llvm.icmp "slt" %3, %1 : i64
    llvm.cond_br %4, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%2 : i64)
  ^bb3(%5: i64):  // 2 preds: ^bb2, ^bb4
    %6 = llvm.icmp "slt" %5, %3 : i64
    llvm.cond_br %6, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @cholesky_kernel_0(%arg0, %3, %5) : (!llvm.ptr, i64, i64) -> ()
    %7 = llvm.mul %5, %1  : i64
    %8 = llvm.add %7, %5  : i64
    %9 = llvm.getelementptr %arg0[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %10 = llvm.load %9 : !llvm.ptr -> f32
    %11 = llvm.mul %3, %1  : i64
    %12 = llvm.add %11, %5  : i64
    %13 = llvm.getelementptr %arg0[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %14 = llvm.load %13 : !llvm.ptr -> f32
    %15 = llvm.fdiv %14, %10  : f32
    llvm.store %15, %13 : f32, !llvm.ptr
    %16 = llvm.add %5, %0  : i64
    llvm.br ^bb3(%16 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @cholesky_kernel_1(%arg0, %3) : (!llvm.ptr, i64) -> ()
    %17 = llvm.mul %3, %1  : i64
    %18 = llvm.add %17, %3  : i64
    %19 = llvm.getelementptr %arg0[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %20 = llvm.load %19 : !llvm.ptr -> f32
    %21 = llvm.intr.sqrt(%20)  : (f32) -> f32
    llvm.store %21, %19 : f32, !llvm.ptr
    %22 = llvm.add %3, %0  : i64
    llvm.br ^bb1(%22 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}

