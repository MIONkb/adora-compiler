// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func private @cholesky_kernel_0(memref<2000x2000xf32>, index, index)

// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func private @cholesky_kernel_1(memref<2000x2000xf32>, index)

// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func @cholesky(%arg0: memref<2000x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
  affine.for %arg1 = 0 to 2000 {
    affine.for %arg2 = 0 to affine_map<(d0) -> (d0)>(%arg1) {
      func.call @cholesky_kernel_0(%arg0, %arg1, %arg2) : (memref<2000x2000xf32>, index, index) -> ()
      %2 = affine.load %arg0[%arg2, %arg2] : memref<2000x2000xf32>
      %3 = affine.load %arg0[%arg1, %arg2] : memref<2000x2000xf32>
      %4 = arith.divf %3, %2 : f32
      affine.store %4, %arg0[%arg1, %arg2] : memref<2000x2000xf32>
    }
    func.call @cholesky_kernel_1(%arg0, %arg1) : (memref<2000x2000xf32>, index) -> ()
    %0 = affine.load %arg0[%arg1, %arg1] : memref<2000x2000xf32>
    %1 = math.sqrt %0 : f32
    affine.store %1, %arg0[%arg1, %arg1] : memref<2000x2000xf32>
  }
  return
}

// -----// IR Dump After ArithExpandOps (arith-expand) //----- //
#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @cholesky_kernel_0(memref<2000x2000xf32>, index, index)
  func.func private @cholesky_kernel_1(memref<2000x2000xf32>, index)
  func.func @cholesky(%arg0: memref<2000x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg1 = 0 to 2000 {
      affine.for %arg2 = 0 to #map(%arg1) {
        func.call @cholesky_kernel_0(%arg0, %arg1, %arg2) : (memref<2000x2000xf32>, index, index) -> ()
        %2 = affine.load %arg0[%arg2, %arg2] : memref<2000x2000xf32>
        %3 = affine.load %arg0[%arg1, %arg2] : memref<2000x2000xf32>
        %4 = arith.divf %3, %2 : f32
        affine.store %4, %arg0[%arg1, %arg2] : memref<2000x2000xf32>
      }
      func.call @cholesky_kernel_1(%arg0, %arg1) : (memref<2000x2000xf32>, index) -> ()
      %0 = affine.load %arg0[%arg1, %arg1] : memref<2000x2000xf32>
      %1 = math.sqrt %0 : f32
      affine.store %1, %arg0[%arg1, %arg1] : memref<2000x2000xf32>
    }
    return
  }
}


// -----// IR Dump After ExpandOps (memref-expand) //----- //
#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @cholesky_kernel_0(memref<2000x2000xf32>, index, index)
  func.func private @cholesky_kernel_1(memref<2000x2000xf32>, index)
  func.func @cholesky(%arg0: memref<2000x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg1 = 0 to 2000 {
      affine.for %arg2 = 0 to #map(%arg1) {
        func.call @cholesky_kernel_0(%arg0, %arg1, %arg2) : (memref<2000x2000xf32>, index, index) -> ()
        %2 = affine.load %arg0[%arg2, %arg2] : memref<2000x2000xf32>
        %3 = affine.load %arg0[%arg1, %arg2] : memref<2000x2000xf32>
        %4 = arith.divf %3, %2 : f32
        affine.store %4, %arg0[%arg1, %arg2] : memref<2000x2000xf32>
      }
      func.call @cholesky_kernel_1(%arg0, %arg1) : (memref<2000x2000xf32>, index) -> ()
      %0 = affine.load %arg0[%arg1, %arg1] : memref<2000x2000xf32>
      %1 = math.sqrt %0 : f32
      affine.store %1, %arg0[%arg1, %arg1] : memref<2000x2000xf32>
    }
    return
  }
}


// -----// IR Dump After NormalizeMemRefs (normalize-memrefs) //----- //
#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @cholesky_kernel_0(memref<2000x2000xf32>, index, index)
  func.func private @cholesky_kernel_1(memref<2000x2000xf32>, index)
  func.func @cholesky(%arg0: memref<2000x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg1 = 0 to 2000 {
      affine.for %arg2 = 0 to #map(%arg1) {
        func.call @cholesky_kernel_0(%arg0, %arg1, %arg2) : (memref<2000x2000xf32>, index, index) -> ()
        %2 = affine.load %arg0[%arg2, %arg2] : memref<2000x2000xf32>
        %3 = affine.load %arg0[%arg1, %arg2] : memref<2000x2000xf32>
        %4 = arith.divf %3, %2 : f32
        affine.store %4, %arg0[%arg1, %arg2] : memref<2000x2000xf32>
      }
      func.call @cholesky_kernel_1(%arg0, %arg1) : (memref<2000x2000xf32>, index) -> ()
      %0 = affine.load %arg0[%arg1, %arg1] : memref<2000x2000xf32>
      %1 = math.sqrt %0 : f32
      affine.store %1, %arg0[%arg1, %arg1] : memref<2000x2000xf32>
    }
    return
  }
}


// -----// IR Dump After ExpandStridedMetadata (expand-strided-metadata) //----- //
#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @cholesky_kernel_0(memref<2000x2000xf32>, index, index)
  func.func private @cholesky_kernel_1(memref<2000x2000xf32>, index)
  func.func @cholesky(%arg0: memref<2000x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg1 = 0 to 2000 {
      affine.for %arg2 = 0 to #map(%arg1) {
        func.call @cholesky_kernel_0(%arg0, %arg1, %arg2) : (memref<2000x2000xf32>, index, index) -> ()
        %2 = affine.load %arg0[%arg2, %arg2] : memref<2000x2000xf32>
        %3 = affine.load %arg0[%arg1, %arg2] : memref<2000x2000xf32>
        %4 = arith.divf %3, %2 : f32
        affine.store %4, %arg0[%arg1, %arg2] : memref<2000x2000xf32>
      }
      func.call @cholesky_kernel_1(%arg0, %arg1) : (memref<2000x2000xf32>, index) -> ()
      %0 = affine.load %arg0[%arg1, %arg1] : memref<2000x2000xf32>
      %1 = math.sqrt %0 : f32
      affine.store %1, %arg0[%arg1, %arg1] : memref<2000x2000xf32>
    }
    return
  }
}


// -----// IR Dump After ConvertAffineToStandard (lower-affine) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @cholesky_kernel_0(memref<2000x2000xf32>, index, index)
  func.func private @cholesky_kernel_1(memref<2000x2000xf32>, index)
  func.func @cholesky(%arg0: memref<2000x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c2000 = arith.constant 2000 : index
    %c1 = arith.constant 1 : index
    scf.for %arg1 = %c0 to %c2000 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      scf.for %arg2 = %c0_0 to %arg1 step %c1_1 {
        func.call @cholesky_kernel_0(%arg0, %arg1, %arg2) : (memref<2000x2000xf32>, index, index) -> ()
        %2 = memref.load %arg0[%arg2, %arg2] : memref<2000x2000xf32>
        %3 = memref.load %arg0[%arg1, %arg2] : memref<2000x2000xf32>
        %4 = arith.divf %3, %2 : f32
        memref.store %4, %arg0[%arg1, %arg2] : memref<2000x2000xf32>
      }
      func.call @cholesky_kernel_1(%arg0, %arg1) : (memref<2000x2000xf32>, index) -> ()
      %0 = memref.load %arg0[%arg1, %arg1] : memref<2000x2000xf32>
      %1 = math.sqrt %0 : f32
      memref.store %1, %arg0[%arg1, %arg1] : memref<2000x2000xf32>
    }
    return
  }
}


// -----// IR Dump After SCFForLoopCanonicalization (scf-for-loop-canonicalization) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @cholesky_kernel_0(memref<2000x2000xf32>, index, index)
  func.func private @cholesky_kernel_1(memref<2000x2000xf32>, index)
  func.func @cholesky(%arg0: memref<2000x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c2000 = arith.constant 2000 : index
    %c1 = arith.constant 1 : index
    scf.for %arg1 = %c0 to %c2000 step %c1 {
      scf.for %arg2 = %c0 to %arg1 step %c1 {
        func.call @cholesky_kernel_0(%arg0, %arg1, %arg2) : (memref<2000x2000xf32>, index, index) -> ()
        %2 = memref.load %arg0[%arg2, %arg2] : memref<2000x2000xf32>
        %3 = memref.load %arg0[%arg1, %arg2] : memref<2000x2000xf32>
        %4 = arith.divf %3, %2 : f32
        memref.store %4, %arg0[%arg1, %arg2] : memref<2000x2000xf32>
      }
      func.call @cholesky_kernel_1(%arg0, %arg1) : (memref<2000x2000xf32>, index) -> ()
      %0 = memref.load %arg0[%arg1, %arg1] : memref<2000x2000xf32>
      %1 = math.sqrt %0 : f32
      memref.store %1, %arg0[%arg1, %arg1] : memref<2000x2000xf32>
    }
    return
  }
}


// -----// IR Dump After SCFToControlFlow (convert-scf-to-cf) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @cholesky_kernel_0(memref<2000x2000xf32>, index, index)
  func.func private @cholesky_kernel_1(memref<2000x2000xf32>, index)
  func.func @cholesky(%arg0: memref<2000x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c2000 = arith.constant 2000 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
    %1 = arith.cmpi slt, %0, %c2000 : index
    cf.cond_br %1, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
    %3 = arith.cmpi slt, %2, %0 : index
    cf.cond_br %3, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    call @cholesky_kernel_0(%arg0, %0, %2) : (memref<2000x2000xf32>, index, index) -> ()
    %4 = memref.load %arg0[%2, %2] : memref<2000x2000xf32>
    %5 = memref.load %arg0[%0, %2] : memref<2000x2000xf32>
    %6 = arith.divf %5, %4 : f32
    memref.store %6, %arg0[%0, %2] : memref<2000x2000xf32>
    %7 = arith.addi %2, %c1 : index
    cf.br ^bb3(%7 : index)
  ^bb5:  // pred: ^bb3
    call @cholesky_kernel_1(%arg0, %0) : (memref<2000x2000xf32>, index) -> ()
    %8 = memref.load %arg0[%0, %0] : memref<2000x2000xf32>
    %9 = math.sqrt %8 : f32
    memref.store %9, %arg0[%0, %0] : memref<2000x2000xf32>
    %10 = arith.addi %0, %c1 : index
    cf.br ^bb1(%10 : index)
  ^bb6:  // pred: ^bb1
    return
  }
}


// -----// IR Dump After ConvertMathToLLVMPass (convert-math-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @cholesky_kernel_0(memref<2000x2000xf32>, index, index)
  func.func private @cholesky_kernel_1(memref<2000x2000xf32>, index)
  func.func @cholesky(%arg0: memref<2000x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c2000 = arith.constant 2000 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
    %1 = arith.cmpi slt, %0, %c2000 : index
    cf.cond_br %1, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
    %3 = arith.cmpi slt, %2, %0 : index
    cf.cond_br %3, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    call @cholesky_kernel_0(%arg0, %0, %2) : (memref<2000x2000xf32>, index, index) -> ()
    %4 = memref.load %arg0[%2, %2] : memref<2000x2000xf32>
    %5 = memref.load %arg0[%0, %2] : memref<2000x2000xf32>
    %6 = arith.divf %5, %4 : f32
    memref.store %6, %arg0[%0, %2] : memref<2000x2000xf32>
    %7 = arith.addi %2, %c1 : index
    cf.br ^bb3(%7 : index)
  ^bb5:  // pred: ^bb3
    call @cholesky_kernel_1(%arg0, %0) : (memref<2000x2000xf32>, index) -> ()
    %8 = memref.load %arg0[%0, %0] : memref<2000x2000xf32>
    %9 = llvm.intr.sqrt(%8)  : (f32) -> f32
    memref.store %9, %arg0[%0, %0] : memref<2000x2000xf32>
    %10 = arith.addi %0, %c1 : index
    cf.br ^bb1(%10 : index)
  ^bb6:  // pred: ^bb1
    return
  }
}


// -----// IR Dump After ConvertMathToLibm (convert-math-to-libm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @cholesky_kernel_0(memref<2000x2000xf32>, index, index)
  func.func private @cholesky_kernel_1(memref<2000x2000xf32>, index)
  func.func @cholesky(%arg0: memref<2000x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c2000 = arith.constant 2000 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
    %1 = arith.cmpi slt, %0, %c2000 : index
    cf.cond_br %1, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
    %3 = arith.cmpi slt, %2, %0 : index
    cf.cond_br %3, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    call @cholesky_kernel_0(%arg0, %0, %2) : (memref<2000x2000xf32>, index, index) -> ()
    %4 = memref.load %arg0[%2, %2] : memref<2000x2000xf32>
    %5 = memref.load %arg0[%0, %2] : memref<2000x2000xf32>
    %6 = arith.divf %5, %4 : f32
    memref.store %6, %arg0[%0, %2] : memref<2000x2000xf32>
    %7 = arith.addi %2, %c1 : index
    cf.br ^bb3(%7 : index)
  ^bb5:  // pred: ^bb3
    call @cholesky_kernel_1(%arg0, %0) : (memref<2000x2000xf32>, index) -> ()
    %8 = memref.load %arg0[%0, %0] : memref<2000x2000xf32>
    %9 = llvm.intr.sqrt(%8)  : (f32) -> f32
    memref.store %9, %arg0[%0, %0] : memref<2000x2000xf32>
    %10 = arith.addi %0, %c1 : index
    cf.br ^bb1(%10 : index)
  ^bb6:  // pred: ^bb1
    return
  }
}


// -----// IR Dump After ArithToLLVMConversionPass (convert-arith-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @cholesky_kernel_0(memref<2000x2000xf32>, index, index)
  func.func private @cholesky_kernel_1(memref<2000x2000xf32>, index)
  func.func @cholesky(%arg0: memref<2000x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(2000 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    cf.br ^bb1(%1 : index)
  ^bb1(%4: index):  // 2 preds: ^bb0, ^bb5
    %5 = builtin.unrealized_conversion_cast %4 : index to i64
    %6 = llvm.icmp "slt" %5, %2 : i64
    cf.cond_br %6, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%1 : index)
  ^bb3(%7: index):  // 2 preds: ^bb2, ^bb4
    %8 = builtin.unrealized_conversion_cast %7 : index to i64
    %9 = llvm.icmp "slt" %8, %5 : i64
    cf.cond_br %9, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    call @cholesky_kernel_0(%arg0, %4, %7) : (memref<2000x2000xf32>, index, index) -> ()
    %10 = memref.load %arg0[%7, %7] : memref<2000x2000xf32>
    %11 = memref.load %arg0[%4, %7] : memref<2000x2000xf32>
    %12 = llvm.fdiv %11, %10  : f32
    memref.store %12, %arg0[%4, %7] : memref<2000x2000xf32>
    %13 = llvm.add %8, %3  : i64
    %14 = builtin.unrealized_conversion_cast %13 : i64 to index
    cf.br ^bb3(%14 : index)
  ^bb5:  // pred: ^bb3
    call @cholesky_kernel_1(%arg0, %4) : (memref<2000x2000xf32>, index) -> ()
    %15 = memref.load %arg0[%4, %4] : memref<2000x2000xf32>
    %16 = llvm.intr.sqrt(%15)  : (f32) -> f32
    memref.store %16, %arg0[%4, %4] : memref<2000x2000xf32>
    %17 = llvm.add %5, %3  : i64
    %18 = builtin.unrealized_conversion_cast %17 : i64 to index
    cf.br ^bb1(%18 : index)
  ^bb6:  // pred: ^bb1
    return
  }
}


// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @cholesky_kernel_0(memref<2000x2000xf32>, index, index)
  func.func private @cholesky_kernel_1(memref<2000x2000xf32>, index)
  func.func @cholesky(%arg0: memref<2000x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<2000x2000xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
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
    call @cholesky_kernel_0(%arg0, %5, %9) : (memref<2000x2000xf32>, index, index) -> ()
    %13 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.mlir.constant(2000 : index) : i64
    %15 = llvm.mul %10, %14  : i64
    %16 = llvm.add %15, %10  : i64
    %17 = llvm.getelementptr %13[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %18 = llvm.load %17 : !llvm.ptr -> f32
    %19 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.mlir.constant(2000 : index) : i64
    %21 = llvm.mul %6, %20  : i64
    %22 = llvm.add %21, %10  : i64
    %23 = llvm.getelementptr %19[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %24 = llvm.load %23 : !llvm.ptr -> f32
    %25 = llvm.fdiv %24, %18  : f32
    %26 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.mlir.constant(2000 : index) : i64
    %28 = llvm.mul %6, %27  : i64
    %29 = llvm.add %28, %10  : i64
    %30 = llvm.getelementptr %26[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %25, %30 : f32, !llvm.ptr
    %31 = llvm.add %11, %4  : i64
    %32 = builtin.unrealized_conversion_cast %31 : i64 to index
    cf.br ^bb3(%32 : index)
  ^bb5:  // pred: ^bb3
    call @cholesky_kernel_1(%arg0, %5) : (memref<2000x2000xf32>, index) -> ()
    %33 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(2000 : index) : i64
    %35 = llvm.mul %6, %34  : i64
    %36 = llvm.add %35, %6  : i64
    %37 = llvm.getelementptr %33[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.load %37 : !llvm.ptr -> f32
    %39 = llvm.intr.sqrt(%38)  : (f32) -> f32
    %40 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.mlir.constant(2000 : index) : i64
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
}


// -----// IR Dump After NormalizeMemRefs (normalize-memrefs) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @cholesky_kernel_0(memref<2000x2000xf32>, index, index)
  func.func private @cholesky_kernel_1(memref<2000x2000xf32>, index)
  func.func @cholesky(%arg0: memref<2000x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<2000x2000xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
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
    call @cholesky_kernel_0(%arg0, %5, %9) : (memref<2000x2000xf32>, index, index) -> ()
    %13 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.mlir.constant(2000 : index) : i64
    %15 = llvm.mul %10, %14  : i64
    %16 = llvm.add %15, %10  : i64
    %17 = llvm.getelementptr %13[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %18 = llvm.load %17 : !llvm.ptr -> f32
    %19 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.mlir.constant(2000 : index) : i64
    %21 = llvm.mul %6, %20  : i64
    %22 = llvm.add %21, %10  : i64
    %23 = llvm.getelementptr %19[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %24 = llvm.load %23 : !llvm.ptr -> f32
    %25 = llvm.fdiv %24, %18  : f32
    %26 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.mlir.constant(2000 : index) : i64
    %28 = llvm.mul %6, %27  : i64
    %29 = llvm.add %28, %10  : i64
    %30 = llvm.getelementptr %26[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %25, %30 : f32, !llvm.ptr
    %31 = llvm.add %11, %4  : i64
    %32 = builtin.unrealized_conversion_cast %31 : i64 to index
    cf.br ^bb3(%32 : index)
  ^bb5:  // pred: ^bb3
    call @cholesky_kernel_1(%arg0, %5) : (memref<2000x2000xf32>, index) -> ()
    %33 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(2000 : index) : i64
    %35 = llvm.mul %6, %34  : i64
    %36 = llvm.add %35, %6  : i64
    %37 = llvm.getelementptr %33[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.load %37 : !llvm.ptr -> f32
    %39 = llvm.intr.sqrt(%38)  : (f32) -> f32
    %40 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.mlir.constant(2000 : index) : i64
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
}


// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @cholesky_kernel_0(memref<2000x2000xf32>, index, index)
  func.func private @cholesky_kernel_1(memref<2000x2000xf32>, index)
  func.func @cholesky(%arg0: memref<2000x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<2000x2000xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
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
    call @cholesky_kernel_0(%arg0, %5, %9) : (memref<2000x2000xf32>, index, index) -> ()
    %13 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.mlir.constant(2000 : index) : i64
    %15 = llvm.mul %10, %14  : i64
    %16 = llvm.add %15, %10  : i64
    %17 = llvm.getelementptr %13[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %18 = llvm.load %17 : !llvm.ptr -> f32
    %19 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.mlir.constant(2000 : index) : i64
    %21 = llvm.mul %6, %20  : i64
    %22 = llvm.add %21, %10  : i64
    %23 = llvm.getelementptr %19[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %24 = llvm.load %23 : !llvm.ptr -> f32
    %25 = llvm.fdiv %24, %18  : f32
    %26 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.mlir.constant(2000 : index) : i64
    %28 = llvm.mul %6, %27  : i64
    %29 = llvm.add %28, %10  : i64
    %30 = llvm.getelementptr %26[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %25, %30 : f32, !llvm.ptr
    %31 = llvm.add %11, %4  : i64
    %32 = builtin.unrealized_conversion_cast %31 : i64 to index
    cf.br ^bb3(%32 : index)
  ^bb5:  // pred: ^bb3
    call @cholesky_kernel_1(%arg0, %5) : (memref<2000x2000xf32>, index) -> ()
    %33 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(2000 : index) : i64
    %35 = llvm.mul %6, %34  : i64
    %36 = llvm.add %35, %6  : i64
    %37 = llvm.getelementptr %33[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.load %37 : !llvm.ptr -> f32
    %39 = llvm.intr.sqrt(%38)  : (f32) -> f32
    %40 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.mlir.constant(2000 : index) : i64
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
}


// -----// IR Dump After ConvertFuncToLLVMPass (convert-func-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @cholesky_kernel_0(!llvm.ptr, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @cholesky_kernel_1(!llvm.ptr, i64) attributes {sym_visibility = "private"}
  llvm.func @cholesky(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.mlir.constant(2000 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.mlir.constant(2000 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.mlir.constant(2000 : index) : i64
    %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = builtin.unrealized_conversion_cast %12 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<2000x2000xf32>
    %14 = builtin.unrealized_conversion_cast %13 : memref<2000x2000xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = builtin.unrealized_conversion_cast %15 : i64 to index
    %17 = llvm.mlir.constant(2000 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%15 : i64)
  ^bb1(%19: i64):  // 2 preds: ^bb0, ^bb5
    %20 = builtin.unrealized_conversion_cast %19 : i64 to index
    %21 = builtin.unrealized_conversion_cast %20 : index to i64
    %22 = builtin.unrealized_conversion_cast %20 : index to i64
    %23 = llvm.icmp "slt" %22, %17 : i64
    llvm.cond_br %23, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%15 : i64)
  ^bb3(%24: i64):  // 2 preds: ^bb2, ^bb4
    %25 = builtin.unrealized_conversion_cast %24 : i64 to index
    %26 = builtin.unrealized_conversion_cast %25 : index to i64
    %27 = builtin.unrealized_conversion_cast %25 : index to i64
    %28 = llvm.icmp "slt" %27, %22 : i64
    llvm.cond_br %28, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %29 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @cholesky_kernel_0(%29, %19, %24) : (!llvm.ptr, i64, i64) -> ()
    %30 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.mlir.constant(2000 : index) : i64
    %32 = llvm.mul %26, %31  : i64
    %33 = llvm.add %32, %26  : i64
    %34 = llvm.getelementptr %30[%33] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %35 = llvm.load %34 : !llvm.ptr -> f32
    %36 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.mlir.constant(2000 : index) : i64
    %38 = llvm.mul %21, %37  : i64
    %39 = llvm.add %38, %26  : i64
    %40 = llvm.getelementptr %36[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %41 = llvm.load %40 : !llvm.ptr -> f32
    %42 = llvm.fdiv %41, %35  : f32
    %43 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.mlir.constant(2000 : index) : i64
    %45 = llvm.mul %21, %44  : i64
    %46 = llvm.add %45, %26  : i64
    %47 = llvm.getelementptr %43[%46] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %42, %47 : f32, !llvm.ptr
    %48 = llvm.add %27, %18  : i64
    %49 = builtin.unrealized_conversion_cast %48 : i64 to index
    llvm.br ^bb3(%48 : i64)
  ^bb5:  // pred: ^bb3
    %50 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @cholesky_kernel_1(%50, %19) : (!llvm.ptr, i64) -> ()
    %51 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.mlir.constant(2000 : index) : i64
    %53 = llvm.mul %21, %52  : i64
    %54 = llvm.add %53, %21  : i64
    %55 = llvm.getelementptr %51[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %56 = llvm.load %55 : !llvm.ptr -> f32
    %57 = llvm.intr.sqrt(%56)  : (f32) -> f32
    %58 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.mlir.constant(2000 : index) : i64
    %60 = llvm.mul %21, %59  : i64
    %61 = llvm.add %60, %21  : i64
    %62 = llvm.getelementptr %58[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %57, %62 : f32, !llvm.ptr
    %63 = llvm.add %22, %18  : i64
    %64 = builtin.unrealized_conversion_cast %63 : i64 to index
    llvm.br ^bb1(%63 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}


// -----// IR Dump After ConvertFuncToLLVMPass (convert-func-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @cholesky_kernel_0(!llvm.ptr, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @cholesky_kernel_1(!llvm.ptr, i64) attributes {sym_visibility = "private"}
  llvm.func @cholesky(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.mlir.constant(2000 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.mlir.constant(2000 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.mlir.constant(2000 : index) : i64
    %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = builtin.unrealized_conversion_cast %12 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<2000x2000xf32>
    %14 = builtin.unrealized_conversion_cast %13 : memref<2000x2000xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = builtin.unrealized_conversion_cast %15 : i64 to index
    %17 = llvm.mlir.constant(2000 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%15 : i64)
  ^bb1(%19: i64):  // 2 preds: ^bb0, ^bb5
    %20 = builtin.unrealized_conversion_cast %19 : i64 to index
    %21 = builtin.unrealized_conversion_cast %20 : index to i64
    %22 = builtin.unrealized_conversion_cast %20 : index to i64
    %23 = llvm.icmp "slt" %22, %17 : i64
    llvm.cond_br %23, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%15 : i64)
  ^bb3(%24: i64):  // 2 preds: ^bb2, ^bb4
    %25 = builtin.unrealized_conversion_cast %24 : i64 to index
    %26 = builtin.unrealized_conversion_cast %25 : index to i64
    %27 = builtin.unrealized_conversion_cast %25 : index to i64
    %28 = llvm.icmp "slt" %27, %22 : i64
    llvm.cond_br %28, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %29 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @cholesky_kernel_0(%29, %19, %24) : (!llvm.ptr, i64, i64) -> ()
    %30 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.mlir.constant(2000 : index) : i64
    %32 = llvm.mul %26, %31  : i64
    %33 = llvm.add %32, %26  : i64
    %34 = llvm.getelementptr %30[%33] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %35 = llvm.load %34 : !llvm.ptr -> f32
    %36 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.mlir.constant(2000 : index) : i64
    %38 = llvm.mul %21, %37  : i64
    %39 = llvm.add %38, %26  : i64
    %40 = llvm.getelementptr %36[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %41 = llvm.load %40 : !llvm.ptr -> f32
    %42 = llvm.fdiv %41, %35  : f32
    %43 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.mlir.constant(2000 : index) : i64
    %45 = llvm.mul %21, %44  : i64
    %46 = llvm.add %45, %26  : i64
    %47 = llvm.getelementptr %43[%46] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %42, %47 : f32, !llvm.ptr
    %48 = llvm.add %27, %18  : i64
    %49 = builtin.unrealized_conversion_cast %48 : i64 to index
    llvm.br ^bb3(%48 : i64)
  ^bb5:  // pred: ^bb3
    %50 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @cholesky_kernel_1(%50, %19) : (!llvm.ptr, i64) -> ()
    %51 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.mlir.constant(2000 : index) : i64
    %53 = llvm.mul %21, %52  : i64
    %54 = llvm.add %53, %21  : i64
    %55 = llvm.getelementptr %51[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %56 = llvm.load %55 : !llvm.ptr -> f32
    %57 = llvm.intr.sqrt(%56)  : (f32) -> f32
    %58 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.mlir.constant(2000 : index) : i64
    %60 = llvm.mul %21, %59  : i64
    %61 = llvm.add %60, %21  : i64
    %62 = llvm.getelementptr %58[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %57, %62 : f32, !llvm.ptr
    %63 = llvm.add %22, %18  : i64
    %64 = builtin.unrealized_conversion_cast %63 : i64 to index
    llvm.br ^bb1(%63 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}


// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @cholesky_kernel_0(!llvm.ptr, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @cholesky_kernel_1(!llvm.ptr, i64) attributes {sym_visibility = "private"}
  llvm.func @cholesky(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.mlir.constant(2000 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.mlir.constant(2000 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.mlir.constant(2000 : index) : i64
    %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = builtin.unrealized_conversion_cast %12 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<2000x2000xf32>
    %14 = builtin.unrealized_conversion_cast %13 : memref<2000x2000xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = builtin.unrealized_conversion_cast %15 : i64 to index
    %17 = llvm.mlir.constant(2000 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%15 : i64)
  ^bb1(%19: i64):  // 2 preds: ^bb0, ^bb5
    %20 = builtin.unrealized_conversion_cast %19 : i64 to index
    %21 = builtin.unrealized_conversion_cast %20 : index to i64
    %22 = builtin.unrealized_conversion_cast %20 : index to i64
    %23 = llvm.icmp "slt" %22, %17 : i64
    llvm.cond_br %23, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%15 : i64)
  ^bb3(%24: i64):  // 2 preds: ^bb2, ^bb4
    %25 = builtin.unrealized_conversion_cast %24 : i64 to index
    %26 = builtin.unrealized_conversion_cast %25 : index to i64
    %27 = builtin.unrealized_conversion_cast %25 : index to i64
    %28 = llvm.icmp "slt" %27, %22 : i64
    llvm.cond_br %28, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %29 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @cholesky_kernel_0(%29, %19, %24) : (!llvm.ptr, i64, i64) -> ()
    %30 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.mlir.constant(2000 : index) : i64
    %32 = llvm.mul %26, %31  : i64
    %33 = llvm.add %32, %26  : i64
    %34 = llvm.getelementptr %30[%33] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %35 = llvm.load %34 : !llvm.ptr -> f32
    %36 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.mlir.constant(2000 : index) : i64
    %38 = llvm.mul %21, %37  : i64
    %39 = llvm.add %38, %26  : i64
    %40 = llvm.getelementptr %36[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %41 = llvm.load %40 : !llvm.ptr -> f32
    %42 = llvm.fdiv %41, %35  : f32
    %43 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.mlir.constant(2000 : index) : i64
    %45 = llvm.mul %21, %44  : i64
    %46 = llvm.add %45, %26  : i64
    %47 = llvm.getelementptr %43[%46] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %42, %47 : f32, !llvm.ptr
    %48 = llvm.add %27, %18  : i64
    %49 = builtin.unrealized_conversion_cast %48 : i64 to index
    llvm.br ^bb3(%48 : i64)
  ^bb5:  // pred: ^bb3
    %50 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @cholesky_kernel_1(%50, %19) : (!llvm.ptr, i64) -> ()
    %51 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.mlir.constant(2000 : index) : i64
    %53 = llvm.mul %21, %52  : i64
    %54 = llvm.add %53, %21  : i64
    %55 = llvm.getelementptr %51[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %56 = llvm.load %55 : !llvm.ptr -> f32
    %57 = llvm.intr.sqrt(%56)  : (f32) -> f32
    %58 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.mlir.constant(2000 : index) : i64
    %60 = llvm.mul %21, %59  : i64
    %61 = llvm.add %60, %21  : i64
    %62 = llvm.getelementptr %58[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %57, %62 : f32, !llvm.ptr
    %63 = llvm.add %22, %18  : i64
    %64 = builtin.unrealized_conversion_cast %63 : i64 to index
    llvm.br ^bb1(%63 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}


// -----// IR Dump After CSE (cse) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @cholesky_kernel_0(!llvm.ptr, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @cholesky_kernel_1(!llvm.ptr, i64) attributes {sym_visibility = "private"}
  llvm.func @cholesky(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.mlir.constant(2000 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %5, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.insertvalue %5, %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.insertvalue %9, %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = builtin.unrealized_conversion_cast %10 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<2000x2000xf32>
    %12 = builtin.unrealized_conversion_cast %11 : memref<2000x2000xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb1(%3 : i64)
  ^bb1(%13: i64):  // 2 preds: ^bb0, ^bb5
    %14 = builtin.unrealized_conversion_cast %13 : i64 to index
    %15 = builtin.unrealized_conversion_cast %14 : index to i64
    %16 = llvm.icmp "slt" %15, %5 : i64
    llvm.cond_br %16, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%3 : i64)
  ^bb3(%17: i64):  // 2 preds: ^bb2, ^bb4
    %18 = builtin.unrealized_conversion_cast %17 : i64 to index
    %19 = builtin.unrealized_conversion_cast %18 : index to i64
    %20 = llvm.icmp "slt" %19, %15 : i64
    llvm.cond_br %20, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %21 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @cholesky_kernel_0(%21, %13, %17) : (!llvm.ptr, i64, i64) -> ()
    %22 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.mul %19, %5  : i64
    %24 = llvm.add %23, %19  : i64
    %25 = llvm.getelementptr %22[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %26 = llvm.load %25 : !llvm.ptr -> f32
    %27 = llvm.mul %15, %5  : i64
    %28 = llvm.add %27, %19  : i64
    %29 = llvm.getelementptr %22[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %30 = llvm.load %29 : !llvm.ptr -> f32
    %31 = llvm.fdiv %30, %26  : f32
    llvm.store %31, %29 : f32, !llvm.ptr
    %32 = llvm.add %19, %9  : i64
    llvm.br ^bb3(%32 : i64)
  ^bb5:  // pred: ^bb3
    %33 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @cholesky_kernel_1(%33, %13) : (!llvm.ptr, i64) -> ()
    %34 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.mul %15, %5  : i64
    %36 = llvm.add %35, %15  : i64
    %37 = llvm.getelementptr %34[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.load %37 : !llvm.ptr -> f32
    %39 = llvm.intr.sqrt(%38)  : (f32) -> f32
    llvm.store %39, %37 : f32, !llvm.ptr
    %40 = llvm.add %15, %9  : i64
    llvm.br ^bb1(%40 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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


// -----// IR Dump After ReconcileUnrealizedCasts (reconcile-unrealized-casts) //----- //
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


