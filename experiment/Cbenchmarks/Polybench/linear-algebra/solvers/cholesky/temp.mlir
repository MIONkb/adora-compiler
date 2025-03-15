// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cholesky(%arg0: i32, %arg1: memref<?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<?x?xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = builtin.unrealized_conversion_cast %1 : i64 to index
    %3 = llvm.mlir.constant(2000 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    cf.br ^bb1(%2 : index)
  ^bb1(%5: index):  // 2 preds: ^bb0, ^bb5
    %6 = builtin.unrealized_conversion_cast %5 : index to i64
    %7 = builtin.unrealized_conversion_cast %5 : index to i64
    %8 = llvm.icmp "slt" %7, %3 : i64
    cf.cond_br %8, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%2 : index)
  ^bb3(%9: index):  // 2 preds: ^bb2, ^bb4
    %10 = builtin.unrealized_conversion_cast %9 : index to i64
    %11 = builtin.unrealized_conversion_cast %9 : index to i64
    %12 = llvm.icmp "slt" %11, %7 : i64
    cf.cond_br %12, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    ADORA.KernelCall @cholesky_kernel_0(%arg1, %5, %9) : (memref<?x?xf32>, index, index) -> ()
    %13 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.mul %10, %14  : i64
    %16 = llvm.add %15, %10  : i64
    %17 = llvm.getelementptr %13[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %18 = llvm.load %17 : !llvm.ptr -> f32
    %19 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.mul %6, %20  : i64
    %22 = llvm.add %21, %10  : i64
    %23 = llvm.getelementptr %19[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %24 = llvm.load %23 : !llvm.ptr -> f32
    %25 = llvm.fdiv %24, %18  : f32
    %26 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.mul %6, %27  : i64
    %29 = llvm.add %28, %10  : i64
    %30 = llvm.getelementptr %26[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %25, %30 : f32, !llvm.ptr
    %31 = llvm.add %11, %4  : i64
    %32 = builtin.unrealized_conversion_cast %31 : i64 to index
    cf.br ^bb3(%32 : index)
  ^bb5:  // pred: ^bb3
    ADORA.KernelCall @cholesky_kernel_1(%arg1, %5) : (memref<?x?xf32>, index) -> ()
    %33 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.mul %6, %34  : i64
    %36 = llvm.add %35, %6  : i64
    %37 = llvm.getelementptr %33[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.load %37 : !llvm.ptr -> f32
    %39 = llvm.intr.sqrt(%38)  : (f32) -> f32
    %40 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.mul %6, %41  : i64
    %43 = llvm.add %42, %6  : i64
    %44 = llvm.getelementptr %40[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %39, %44 : f32, !llvm.ptr
    %45 = llvm.add %7, %4  : i64
    %46 = builtin.unrealized_conversion_cast %45 : i64 to index
    cf.br ^bb1(%46 : index)
  ^bb6:  // pred: ^bb1
    return
  }
  func.func private @cholesky_kernel_0(memref<?x?xf32>, index, index)
  func.func private @cholesky_kernel_1(memref<?x?xf32>, index)
}