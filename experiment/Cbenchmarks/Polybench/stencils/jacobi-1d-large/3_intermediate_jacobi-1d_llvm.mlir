// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func private @jacobi_1d_kernel_1(memref<2000xf32>, memref<2000xf32>)

// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func private @jacobi_1d_kernel_0(memref<2000xf32>, memref<2000xf32>)

// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
  %cst = arith.constant 3.333300e-01 : f32
  affine.for %arg2 = 0 to 500 {
    func.call @jacobi_1d_kernel_0(%arg0, %arg1) : (memref<2000xf32>, memref<2000xf32>) -> ()
    func.call @jacobi_1d_kernel_1(%arg1, %arg0) : (memref<2000xf32>, memref<2000xf32>) -> ()
  }
  return
}

// -----// IR Dump After ArithExpandOps (arith-expand) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.333300e-01 : f32
    affine.for %arg2 = 0 to 500 {
      func.call @jacobi_1d_kernel_0(%arg0, %arg1) : (memref<2000xf32>, memref<2000xf32>) -> ()
      func.call @jacobi_1d_kernel_1(%arg1, %arg0) : (memref<2000xf32>, memref<2000xf32>) -> ()
    }
    return
  }
  func.func private @jacobi_1d_kernel_0(memref<2000xf32>, memref<2000xf32>)
  func.func private @jacobi_1d_kernel_1(memref<2000xf32>, memref<2000xf32>)
}


// -----// IR Dump After ExpandOps (memref-expand) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.333300e-01 : f32
    affine.for %arg2 = 0 to 500 {
      func.call @jacobi_1d_kernel_0(%arg0, %arg1) : (memref<2000xf32>, memref<2000xf32>) -> ()
      func.call @jacobi_1d_kernel_1(%arg1, %arg0) : (memref<2000xf32>, memref<2000xf32>) -> ()
    }
    return
  }
  func.func private @jacobi_1d_kernel_0(memref<2000xf32>, memref<2000xf32>)
  func.func private @jacobi_1d_kernel_1(memref<2000xf32>, memref<2000xf32>)
}


// -----// IR Dump After NormalizeMemRefs (normalize-memrefs) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.333300e-01 : f32
    affine.for %arg2 = 0 to 500 {
      func.call @jacobi_1d_kernel_0(%arg0, %arg1) : (memref<2000xf32>, memref<2000xf32>) -> ()
      func.call @jacobi_1d_kernel_1(%arg1, %arg0) : (memref<2000xf32>, memref<2000xf32>) -> ()
    }
    return
  }
  func.func private @jacobi_1d_kernel_0(memref<2000xf32>, memref<2000xf32>)
  func.func private @jacobi_1d_kernel_1(memref<2000xf32>, memref<2000xf32>)
}


// -----// IR Dump After ExpandStridedMetadata (expand-strided-metadata) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg2 = 0 to 500 {
      func.call @jacobi_1d_kernel_0(%arg0, %arg1) : (memref<2000xf32>, memref<2000xf32>) -> ()
      func.call @jacobi_1d_kernel_1(%arg1, %arg0) : (memref<2000xf32>, memref<2000xf32>) -> ()
    }
    return
  }
  func.func private @jacobi_1d_kernel_0(memref<2000xf32>, memref<2000xf32>)
  func.func private @jacobi_1d_kernel_1(memref<2000xf32>, memref<2000xf32>)
}


// -----// IR Dump After ConvertAffineToStandard (lower-affine) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c500 = arith.constant 500 : index
    %c1 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c500 step %c1 {
      func.call @jacobi_1d_kernel_0(%arg0, %arg1) : (memref<2000xf32>, memref<2000xf32>) -> ()
      func.call @jacobi_1d_kernel_1(%arg1, %arg0) : (memref<2000xf32>, memref<2000xf32>) -> ()
    }
    return
  }
  func.func private @jacobi_1d_kernel_0(memref<2000xf32>, memref<2000xf32>)
  func.func private @jacobi_1d_kernel_1(memref<2000xf32>, memref<2000xf32>)
}


// -----// IR Dump After SCFForLoopCanonicalization (scf-for-loop-canonicalization) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c500 = arith.constant 500 : index
    %c1 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c500 step %c1 {
      func.call @jacobi_1d_kernel_0(%arg0, %arg1) : (memref<2000xf32>, memref<2000xf32>) -> ()
      func.call @jacobi_1d_kernel_1(%arg1, %arg0) : (memref<2000xf32>, memref<2000xf32>) -> ()
    }
    return
  }
  func.func private @jacobi_1d_kernel_0(memref<2000xf32>, memref<2000xf32>)
  func.func private @jacobi_1d_kernel_1(memref<2000xf32>, memref<2000xf32>)
}


// -----// IR Dump After SCFToControlFlow (convert-scf-to-cf) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c500 = arith.constant 500 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
    %1 = arith.cmpi slt, %0, %c500 : index
    cf.cond_br %1, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    call @jacobi_1d_kernel_0(%arg0, %arg1) : (memref<2000xf32>, memref<2000xf32>) -> ()
    call @jacobi_1d_kernel_1(%arg1, %arg0) : (memref<2000xf32>, memref<2000xf32>) -> ()
    %2 = arith.addi %0, %c1 : index
    cf.br ^bb1(%2 : index)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func private @jacobi_1d_kernel_0(memref<2000xf32>, memref<2000xf32>)
  func.func private @jacobi_1d_kernel_1(memref<2000xf32>, memref<2000xf32>)
}


// -----// IR Dump After ConvertMathToLLVMPass (convert-math-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c500 = arith.constant 500 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
    %1 = arith.cmpi slt, %0, %c500 : index
    cf.cond_br %1, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    call @jacobi_1d_kernel_0(%arg0, %arg1) : (memref<2000xf32>, memref<2000xf32>) -> ()
    call @jacobi_1d_kernel_1(%arg1, %arg0) : (memref<2000xf32>, memref<2000xf32>) -> ()
    %2 = arith.addi %0, %c1 : index
    cf.br ^bb1(%2 : index)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func private @jacobi_1d_kernel_0(memref<2000xf32>, memref<2000xf32>)
  func.func private @jacobi_1d_kernel_1(memref<2000xf32>, memref<2000xf32>)
}


// -----// IR Dump After ConvertMathToLibm (convert-math-to-libm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c500 = arith.constant 500 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
    %1 = arith.cmpi slt, %0, %c500 : index
    cf.cond_br %1, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    call @jacobi_1d_kernel_0(%arg0, %arg1) : (memref<2000xf32>, memref<2000xf32>) -> ()
    call @jacobi_1d_kernel_1(%arg1, %arg0) : (memref<2000xf32>, memref<2000xf32>) -> ()
    %2 = arith.addi %0, %c1 : index
    cf.br ^bb1(%2 : index)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func private @jacobi_1d_kernel_0(memref<2000xf32>, memref<2000xf32>)
  func.func private @jacobi_1d_kernel_1(memref<2000xf32>, memref<2000xf32>)
}


// -----// IR Dump After ArithToLLVMConversionPass (convert-arith-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(500 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    cf.br ^bb1(%1 : index)
  ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
    %5 = builtin.unrealized_conversion_cast %4 : index to i64
    %6 = llvm.icmp "slt" %5, %2 : i64
    cf.cond_br %6, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    call @jacobi_1d_kernel_0(%arg0, %arg1) : (memref<2000xf32>, memref<2000xf32>) -> ()
    call @jacobi_1d_kernel_1(%arg1, %arg0) : (memref<2000xf32>, memref<2000xf32>) -> ()
    %7 = llvm.add %5, %3  : i64
    %8 = builtin.unrealized_conversion_cast %7 : i64 to index
    cf.br ^bb1(%8 : index)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func private @jacobi_1d_kernel_0(memref<2000xf32>, memref<2000xf32>)
  func.func private @jacobi_1d_kernel_1(memref<2000xf32>, memref<2000xf32>)
}


// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(500 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    cf.br ^bb1(%1 : index)
  ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
    %5 = builtin.unrealized_conversion_cast %4 : index to i64
    %6 = llvm.icmp "slt" %5, %2 : i64
    cf.cond_br %6, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    call @jacobi_1d_kernel_0(%arg0, %arg1) : (memref<2000xf32>, memref<2000xf32>) -> ()
    call @jacobi_1d_kernel_1(%arg1, %arg0) : (memref<2000xf32>, memref<2000xf32>) -> ()
    %7 = llvm.add %5, %3  : i64
    %8 = builtin.unrealized_conversion_cast %7 : i64 to index
    cf.br ^bb1(%8 : index)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func private @jacobi_1d_kernel_0(memref<2000xf32>, memref<2000xf32>)
  func.func private @jacobi_1d_kernel_1(memref<2000xf32>, memref<2000xf32>)
}


// -----// IR Dump After NormalizeMemRefs (normalize-memrefs) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(500 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    cf.br ^bb1(%1 : index)
  ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
    %5 = builtin.unrealized_conversion_cast %4 : index to i64
    %6 = llvm.icmp "slt" %5, %2 : i64
    cf.cond_br %6, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    call @jacobi_1d_kernel_0(%arg0, %arg1) : (memref<2000xf32>, memref<2000xf32>) -> ()
    call @jacobi_1d_kernel_1(%arg1, %arg0) : (memref<2000xf32>, memref<2000xf32>) -> ()
    %7 = llvm.add %5, %3  : i64
    %8 = builtin.unrealized_conversion_cast %7 : i64 to index
    cf.br ^bb1(%8 : index)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func private @jacobi_1d_kernel_0(memref<2000xf32>, memref<2000xf32>)
  func.func private @jacobi_1d_kernel_1(memref<2000xf32>, memref<2000xf32>)
}


// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(500 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    cf.br ^bb1(%1 : index)
  ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
    %5 = builtin.unrealized_conversion_cast %4 : index to i64
    %6 = llvm.icmp "slt" %5, %2 : i64
    cf.cond_br %6, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    call @jacobi_1d_kernel_0(%arg0, %arg1) : (memref<2000xf32>, memref<2000xf32>) -> ()
    call @jacobi_1d_kernel_1(%arg1, %arg0) : (memref<2000xf32>, memref<2000xf32>) -> ()
    %7 = llvm.add %5, %3  : i64
    %8 = builtin.unrealized_conversion_cast %7 : i64 to index
    cf.br ^bb1(%8 : index)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func private @jacobi_1d_kernel_0(memref<2000xf32>, memref<2000xf32>)
  func.func private @jacobi_1d_kernel_1(memref<2000xf32>, memref<2000xf32>)
}


// -----// IR Dump After ConvertFuncToLLVMPass (convert-func-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @jacobi_1d(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.mlir.constant(2000 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg1, %9[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg1, %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.insertvalue %12, %11[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.mlir.constant(2000 : index) : i64
    %15 = llvm.insertvalue %14, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.insertvalue %16, %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = builtin.unrealized_conversion_cast %18 : i64 to index
    %20 = llvm.mlir.constant(500 : index) : i64
    %21 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%18 : i64)
  ^bb1(%22: i64):  // 2 preds: ^bb0, ^bb2
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    %24 = builtin.unrealized_conversion_cast %23 : index to i64
    %25 = llvm.icmp "slt" %24, %20 : i64
    llvm.cond_br %25, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %26 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @jacobi_1d_kernel_0(%26, %27) : (!llvm.ptr, !llvm.ptr) -> ()
    %28 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @jacobi_1d_kernel_1(%28, %29) : (!llvm.ptr, !llvm.ptr) -> ()
    %30 = llvm.add %24, %21  : i64
    %31 = builtin.unrealized_conversion_cast %30 : i64 to index
    llvm.br ^bb1(%30 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @jacobi_1d_kernel_0(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @jacobi_1d_kernel_1(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}


// -----// IR Dump After ConvertFuncToLLVMPass (convert-func-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @jacobi_1d(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.mlir.constant(2000 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg1, %9[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg1, %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.insertvalue %12, %11[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.mlir.constant(2000 : index) : i64
    %15 = llvm.insertvalue %14, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.insertvalue %16, %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = builtin.unrealized_conversion_cast %18 : i64 to index
    %20 = llvm.mlir.constant(500 : index) : i64
    %21 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%18 : i64)
  ^bb1(%22: i64):  // 2 preds: ^bb0, ^bb2
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    %24 = builtin.unrealized_conversion_cast %23 : index to i64
    %25 = llvm.icmp "slt" %24, %20 : i64
    llvm.cond_br %25, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %26 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @jacobi_1d_kernel_0(%26, %27) : (!llvm.ptr, !llvm.ptr) -> ()
    %28 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @jacobi_1d_kernel_1(%28, %29) : (!llvm.ptr, !llvm.ptr) -> ()
    %30 = llvm.add %24, %21  : i64
    %31 = builtin.unrealized_conversion_cast %30 : i64 to index
    llvm.br ^bb1(%30 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @jacobi_1d_kernel_0(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @jacobi_1d_kernel_1(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}


// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @jacobi_1d(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.mlir.constant(2000 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg1, %9[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg1, %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.insertvalue %12, %11[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.mlir.constant(2000 : index) : i64
    %15 = llvm.insertvalue %14, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.insertvalue %16, %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = builtin.unrealized_conversion_cast %18 : i64 to index
    %20 = llvm.mlir.constant(500 : index) : i64
    %21 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%18 : i64)
  ^bb1(%22: i64):  // 2 preds: ^bb0, ^bb2
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    %24 = builtin.unrealized_conversion_cast %23 : index to i64
    %25 = llvm.icmp "slt" %24, %20 : i64
    llvm.cond_br %25, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %26 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @jacobi_1d_kernel_0(%26, %27) : (!llvm.ptr, !llvm.ptr) -> ()
    %28 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @jacobi_1d_kernel_1(%28, %29) : (!llvm.ptr, !llvm.ptr) -> ()
    %30 = llvm.add %24, %21  : i64
    %31 = builtin.unrealized_conversion_cast %30 : i64 to index
    llvm.br ^bb1(%30 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @jacobi_1d_kernel_0(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @jacobi_1d_kernel_1(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}


// -----// IR Dump After CSE (cse) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @jacobi_1d(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.mlir.constant(2000 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg1, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %3, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %5, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %7, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.mlir.constant(500 : index) : i64
    llvm.br ^bb1(%3 : i64)
  ^bb1(%15: i64):  // 2 preds: ^bb0, ^bb2
    %16 = builtin.unrealized_conversion_cast %15 : i64 to index
    %17 = builtin.unrealized_conversion_cast %16 : index to i64
    %18 = llvm.icmp "slt" %17, %14 : i64
    llvm.cond_br %18, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %19 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.extractvalue %13[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @jacobi_1d_kernel_0(%19, %20) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @jacobi_1d_kernel_1(%20, %19) : (!llvm.ptr, !llvm.ptr) -> ()
    %21 = llvm.add %17, %7  : i64
    llvm.br ^bb1(%21 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @jacobi_1d_kernel_0(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @jacobi_1d_kernel_1(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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


// -----// IR Dump After ReconcileUnrealizedCasts (reconcile-unrealized-casts) //----- //
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


