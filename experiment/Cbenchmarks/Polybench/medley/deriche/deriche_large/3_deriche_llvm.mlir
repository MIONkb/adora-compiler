module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @deriche(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.undef : f32
    %2 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr
    llvm.store %1, %2 : f32, !llvm.ptr
    %3 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr
    llvm.store %1, %3 : f32, !llvm.ptr
    %4 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr
    llvm.store %1, %4 : f32, !llvm.ptr
    %5 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr
    llvm.store %1, %5 : f32, !llvm.ptr
    %6 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr
    llvm.store %1, %6 : f32, !llvm.ptr
    %7 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr
    llvm.store %1, %7 : f32, !llvm.ptr
    %8 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr
    llvm.store %1, %8 : f32, !llvm.ptr
    %9 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr
    llvm.store %1, %9 : f32, !llvm.ptr
    %10 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr
    llvm.store %1, %10 : f32, !llvm.ptr
    %11 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr
    llvm.store %1, %11 : f32, !llvm.ptr
    llvm.call @deriche_kernel_0(%9, %8, %11, %arg0, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @deriche_kernel_1(%3, %2, %7, %6, %arg3, %arg0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @deriche_kernel_2(%arg2, %arg3, %arg1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @deriche_kernel_3(%10, %9, %8, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @deriche_kernel_4(%5, %4, %3, %2, %arg3, %arg1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @deriche_kernel_5(%arg2, %arg3, %arg1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @deriche_kernel_0(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_1(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_2(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_3(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_4(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_5(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}

