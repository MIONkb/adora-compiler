// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)

// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)

// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)

// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)

// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)

// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)

// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
  %cst = arith.constant 0.110209078 : f32
  %cst_0 = arith.constant -0.183681786 : f32
  %cst_1 = arith.constant 0.114441216 : f32
  %cst_2 = arith.constant -0.188681662 : f32
  %cst_3 = arith.constant 0.840896427 : f32
  %cst_4 = arith.constant -0.606530666 : f32
  %cst_5 = arith.constant 0.000000e+00 : f32
  %0 = llvm.mlir.undef : f32
  %alloca = memref.alloca() : memref<f32>
  affine.store %0, %alloca[] : memref<f32>
  %alloca_6 = memref.alloca() : memref<f32>
  affine.store %0, %alloca_6[] : memref<f32>
  %alloca_7 = memref.alloca() : memref<f32>
  affine.store %0, %alloca_7[] : memref<f32>
  %alloca_8 = memref.alloca() : memref<f32>
  affine.store %0, %alloca_8[] : memref<f32>
  %alloca_9 = memref.alloca() : memref<f32>
  affine.store %0, %alloca_9[] : memref<f32>
  %alloca_10 = memref.alloca() : memref<f32>
  affine.store %0, %alloca_10[] : memref<f32>
  %alloca_11 = memref.alloca() : memref<f32>
  affine.store %0, %alloca_11[] : memref<f32>
  %alloca_12 = memref.alloca() : memref<f32>
  affine.store %0, %alloca_12[] : memref<f32>
  %alloca_13 = memref.alloca() : memref<f32>
  affine.store %0, %alloca_13[] : memref<f32>
  %alloca_14 = memref.alloca() : memref<f32>
  affine.store %0, %alloca_14[] : memref<f32>
  call @deriche_kernel_0(%alloca_12, %alloca_11, %alloca_14, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
  call @deriche_kernel_1(%alloca_6, %alloca, %alloca_10, %alloca_9, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
  call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
  call @deriche_kernel_3(%alloca_13, %alloca_12, %alloca_11, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
  call @deriche_kernel_4(%alloca_8, %alloca_7, %alloca_6, %alloca, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
  call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
  return
}

// -----// IR Dump After ArithExpandOps (arith-expand) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.110209078 : f32
    %cst_0 = arith.constant -0.183681786 : f32
    %cst_1 = arith.constant 0.114441216 : f32
    %cst_2 = arith.constant -0.188681662 : f32
    %cst_3 = arith.constant 0.840896427 : f32
    %cst_4 = arith.constant -0.606530666 : f32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %0, %alloca[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_8[] : memref<f32>
    %alloca_9 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_9[] : memref<f32>
    %alloca_10 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_10[] : memref<f32>
    %alloca_11 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_11[] : memref<f32>
    %alloca_12 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_12[] : memref<f32>
    %alloca_13 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_13[] : memref<f32>
    %alloca_14 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_14[] : memref<f32>
    call @deriche_kernel_0(%alloca_12, %alloca_11, %alloca_14, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%alloca_6, %alloca, %alloca_10, %alloca_9, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%alloca_13, %alloca_12, %alloca_11, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%alloca_8, %alloca_7, %alloca_6, %alloca, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}


// -----// IR Dump After ExpandOps (memref-expand) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.110209078 : f32
    %cst_0 = arith.constant -0.183681786 : f32
    %cst_1 = arith.constant 0.114441216 : f32
    %cst_2 = arith.constant -0.188681662 : f32
    %cst_3 = arith.constant 0.840896427 : f32
    %cst_4 = arith.constant -0.606530666 : f32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %0, %alloca[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_8[] : memref<f32>
    %alloca_9 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_9[] : memref<f32>
    %alloca_10 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_10[] : memref<f32>
    %alloca_11 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_11[] : memref<f32>
    %alloca_12 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_12[] : memref<f32>
    %alloca_13 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_13[] : memref<f32>
    %alloca_14 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_14[] : memref<f32>
    call @deriche_kernel_0(%alloca_12, %alloca_11, %alloca_14, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%alloca_6, %alloca, %alloca_10, %alloca_9, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%alloca_13, %alloca_12, %alloca_11, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%alloca_8, %alloca_7, %alloca_6, %alloca, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}


// -----// IR Dump After NormalizeMemRefs (normalize-memrefs) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.110209078 : f32
    %cst_0 = arith.constant -0.183681786 : f32
    %cst_1 = arith.constant 0.114441216 : f32
    %cst_2 = arith.constant -0.188681662 : f32
    %cst_3 = arith.constant 0.840896427 : f32
    %cst_4 = arith.constant -0.606530666 : f32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %0, %alloca[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_8[] : memref<f32>
    %alloca_9 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_9[] : memref<f32>
    %alloca_10 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_10[] : memref<f32>
    %alloca_11 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_11[] : memref<f32>
    %alloca_12 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_12[] : memref<f32>
    %alloca_13 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_13[] : memref<f32>
    %alloca_14 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_14[] : memref<f32>
    call @deriche_kernel_0(%alloca_12, %alloca_11, %alloca_14, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%alloca_6, %alloca, %alloca_10, %alloca_9, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%alloca_13, %alloca_12, %alloca_11, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%alloca_8, %alloca_7, %alloca_6, %alloca, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}


// -----// IR Dump After ExpandStridedMetadata (expand-strided-metadata) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %0, %alloca[] : memref<f32>
    %alloca_0 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_0[] : memref<f32>
    %alloca_1 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_1[] : memref<f32>
    %alloca_2 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_2[] : memref<f32>
    %alloca_3 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_3[] : memref<f32>
    %alloca_4 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_4[] : memref<f32>
    %alloca_5 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_5[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_8[] : memref<f32>
    call @deriche_kernel_0(%alloca_6, %alloca_5, %alloca_8, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%alloca_0, %alloca, %alloca_4, %alloca_3, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%alloca_7, %alloca_6, %alloca_5, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%alloca_2, %alloca_1, %alloca_0, %alloca, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}


// -----// IR Dump After ConvertAffineToStandard (lower-affine) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    memref.store %0, %alloca[] : memref<f32>
    %alloca_0 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_0[] : memref<f32>
    %alloca_1 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_1[] : memref<f32>
    %alloca_2 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_2[] : memref<f32>
    %alloca_3 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_3[] : memref<f32>
    %alloca_4 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_4[] : memref<f32>
    %alloca_5 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_5[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_8[] : memref<f32>
    call @deriche_kernel_0(%alloca_6, %alloca_5, %alloca_8, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%alloca_0, %alloca, %alloca_4, %alloca_3, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%alloca_7, %alloca_6, %alloca_5, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%alloca_2, %alloca_1, %alloca_0, %alloca, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}


// -----// IR Dump After SCFForLoopCanonicalization (scf-for-loop-canonicalization) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    memref.store %0, %alloca[] : memref<f32>
    %alloca_0 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_0[] : memref<f32>
    %alloca_1 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_1[] : memref<f32>
    %alloca_2 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_2[] : memref<f32>
    %alloca_3 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_3[] : memref<f32>
    %alloca_4 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_4[] : memref<f32>
    %alloca_5 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_5[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_8[] : memref<f32>
    call @deriche_kernel_0(%alloca_6, %alloca_5, %alloca_8, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%alloca_0, %alloca, %alloca_4, %alloca_3, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%alloca_7, %alloca_6, %alloca_5, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%alloca_2, %alloca_1, %alloca_0, %alloca, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}


// -----// IR Dump After SCFToControlFlow (convert-scf-to-cf) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    memref.store %0, %alloca[] : memref<f32>
    %alloca_0 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_0[] : memref<f32>
    %alloca_1 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_1[] : memref<f32>
    %alloca_2 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_2[] : memref<f32>
    %alloca_3 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_3[] : memref<f32>
    %alloca_4 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_4[] : memref<f32>
    %alloca_5 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_5[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_8[] : memref<f32>
    call @deriche_kernel_0(%alloca_6, %alloca_5, %alloca_8, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%alloca_0, %alloca, %alloca_4, %alloca_3, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%alloca_7, %alloca_6, %alloca_5, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%alloca_2, %alloca_1, %alloca_0, %alloca, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}


// -----// IR Dump After ConvertMathToLLVMPass (convert-math-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    memref.store %0, %alloca[] : memref<f32>
    %alloca_0 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_0[] : memref<f32>
    %alloca_1 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_1[] : memref<f32>
    %alloca_2 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_2[] : memref<f32>
    %alloca_3 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_3[] : memref<f32>
    %alloca_4 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_4[] : memref<f32>
    %alloca_5 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_5[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_8[] : memref<f32>
    call @deriche_kernel_0(%alloca_6, %alloca_5, %alloca_8, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%alloca_0, %alloca, %alloca_4, %alloca_3, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%alloca_7, %alloca_6, %alloca_5, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%alloca_2, %alloca_1, %alloca_0, %alloca, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}


// -----// IR Dump After ConvertMathToLibm (convert-math-to-libm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    memref.store %0, %alloca[] : memref<f32>
    %alloca_0 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_0[] : memref<f32>
    %alloca_1 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_1[] : memref<f32>
    %alloca_2 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_2[] : memref<f32>
    %alloca_3 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_3[] : memref<f32>
    %alloca_4 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_4[] : memref<f32>
    %alloca_5 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_5[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_8[] : memref<f32>
    call @deriche_kernel_0(%alloca_6, %alloca_5, %alloca_8, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%alloca_0, %alloca, %alloca_4, %alloca_3, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%alloca_7, %alloca_6, %alloca_5, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%alloca_2, %alloca_1, %alloca_0, %alloca, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}


// -----// IR Dump After ArithToLLVMConversionPass (convert-arith-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    memref.store %0, %alloca[] : memref<f32>
    %alloca_0 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_0[] : memref<f32>
    %alloca_1 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_1[] : memref<f32>
    %alloca_2 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_2[] : memref<f32>
    %alloca_3 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_3[] : memref<f32>
    %alloca_4 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_4[] : memref<f32>
    %alloca_5 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_5[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_8[] : memref<f32>
    call @deriche_kernel_0(%alloca_6, %alloca_5, %alloca_8, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%alloca_0, %alloca, %alloca_4, %alloca_3, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%alloca_7, %alloca_6, %alloca_5, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%alloca_2, %alloca_1, %alloca_0, %alloca, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}


// -----// IR Dump After NormalizeMemRefs (normalize-memrefs) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    memref.store %0, %alloca[] : memref<f32>
    %alloca_0 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_0[] : memref<f32>
    %alloca_1 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_1[] : memref<f32>
    %alloca_2 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_2[] : memref<f32>
    %alloca_3 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_3[] : memref<f32>
    %alloca_4 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_4[] : memref<f32>
    %alloca_5 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_5[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    memref.store %0, %alloca_8[] : memref<f32>
    call @deriche_kernel_0(%alloca_6, %alloca_5, %alloca_8, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%alloca_0, %alloca, %alloca_4, %alloca_3, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%alloca_7, %alloca_6, %alloca_5, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%alloca_2, %alloca_1, %alloca_0, %alloca, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}


// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.undef : f32
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.alloca %1 x f32 : (i64) -> !llvm.ptr
    %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<(ptr, ptr, i64)> 
    %5 = llvm.insertvalue %2, %4[1] : !llvm.struct<(ptr, ptr, i64)> 
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.insertvalue %6, %5[2] : !llvm.struct<(ptr, ptr, i64)> 
    %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    %9 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %0, %9 : f32, !llvm.ptr
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.alloca %10 x f32 : (i64) -> !llvm.ptr
    %12 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %13 = llvm.insertvalue %11, %12[0] : !llvm.struct<(ptr, ptr, i64)> 
    %14 = llvm.insertvalue %11, %13[1] : !llvm.struct<(ptr, ptr, i64)> 
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.insertvalue %15, %14[2] : !llvm.struct<(ptr, ptr, i64)> 
    %17 = builtin.unrealized_conversion_cast %16 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    %18 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %0, %18 : f32, !llvm.ptr
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.alloca %19 x f32 : (i64) -> !llvm.ptr
    %21 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<(ptr, ptr, i64)> 
    %23 = llvm.insertvalue %20, %22[1] : !llvm.struct<(ptr, ptr, i64)> 
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = llvm.insertvalue %24, %23[2] : !llvm.struct<(ptr, ptr, i64)> 
    %26 = builtin.unrealized_conversion_cast %25 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    %27 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %0, %27 : f32, !llvm.ptr
    %28 = llvm.mlir.constant(1 : index) : i64
    %29 = llvm.alloca %28 x f32 : (i64) -> !llvm.ptr
    %30 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %31 = llvm.insertvalue %29, %30[0] : !llvm.struct<(ptr, ptr, i64)> 
    %32 = llvm.insertvalue %29, %31[1] : !llvm.struct<(ptr, ptr, i64)> 
    %33 = llvm.mlir.constant(0 : index) : i64
    %34 = llvm.insertvalue %33, %32[2] : !llvm.struct<(ptr, ptr, i64)> 
    %35 = builtin.unrealized_conversion_cast %34 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    %36 = llvm.extractvalue %34[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %0, %36 : f32, !llvm.ptr
    %37 = llvm.mlir.constant(1 : index) : i64
    %38 = llvm.alloca %37 x f32 : (i64) -> !llvm.ptr
    %39 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %40 = llvm.insertvalue %38, %39[0] : !llvm.struct<(ptr, ptr, i64)> 
    %41 = llvm.insertvalue %38, %40[1] : !llvm.struct<(ptr, ptr, i64)> 
    %42 = llvm.mlir.constant(0 : index) : i64
    %43 = llvm.insertvalue %42, %41[2] : !llvm.struct<(ptr, ptr, i64)> 
    %44 = builtin.unrealized_conversion_cast %43 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    %45 = llvm.extractvalue %43[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %0, %45 : f32, !llvm.ptr
    %46 = llvm.mlir.constant(1 : index) : i64
    %47 = llvm.alloca %46 x f32 : (i64) -> !llvm.ptr
    %48 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %49 = llvm.insertvalue %47, %48[0] : !llvm.struct<(ptr, ptr, i64)> 
    %50 = llvm.insertvalue %47, %49[1] : !llvm.struct<(ptr, ptr, i64)> 
    %51 = llvm.mlir.constant(0 : index) : i64
    %52 = llvm.insertvalue %51, %50[2] : !llvm.struct<(ptr, ptr, i64)> 
    %53 = builtin.unrealized_conversion_cast %52 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    %54 = llvm.extractvalue %52[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %0, %54 : f32, !llvm.ptr
    %55 = llvm.mlir.constant(1 : index) : i64
    %56 = llvm.alloca %55 x f32 : (i64) -> !llvm.ptr
    %57 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %58 = llvm.insertvalue %56, %57[0] : !llvm.struct<(ptr, ptr, i64)> 
    %59 = llvm.insertvalue %56, %58[1] : !llvm.struct<(ptr, ptr, i64)> 
    %60 = llvm.mlir.constant(0 : index) : i64
    %61 = llvm.insertvalue %60, %59[2] : !llvm.struct<(ptr, ptr, i64)> 
    %62 = builtin.unrealized_conversion_cast %61 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    %63 = llvm.extractvalue %61[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %0, %63 : f32, !llvm.ptr
    %64 = llvm.mlir.constant(1 : index) : i64
    %65 = llvm.alloca %64 x f32 : (i64) -> !llvm.ptr
    %66 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %67 = llvm.insertvalue %65, %66[0] : !llvm.struct<(ptr, ptr, i64)> 
    %68 = llvm.insertvalue %65, %67[1] : !llvm.struct<(ptr, ptr, i64)> 
    %69 = llvm.mlir.constant(0 : index) : i64
    %70 = llvm.insertvalue %69, %68[2] : !llvm.struct<(ptr, ptr, i64)> 
    %71 = builtin.unrealized_conversion_cast %70 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    %72 = llvm.extractvalue %70[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %0, %72 : f32, !llvm.ptr
    %73 = llvm.mlir.constant(1 : index) : i64
    %74 = llvm.alloca %73 x f32 : (i64) -> !llvm.ptr
    %75 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %76 = llvm.insertvalue %74, %75[0] : !llvm.struct<(ptr, ptr, i64)> 
    %77 = llvm.insertvalue %74, %76[1] : !llvm.struct<(ptr, ptr, i64)> 
    %78 = llvm.mlir.constant(0 : index) : i64
    %79 = llvm.insertvalue %78, %77[2] : !llvm.struct<(ptr, ptr, i64)> 
    %80 = builtin.unrealized_conversion_cast %79 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    %81 = llvm.extractvalue %79[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %0, %81 : f32, !llvm.ptr
    %82 = llvm.mlir.constant(1 : index) : i64
    %83 = llvm.alloca %82 x f32 : (i64) -> !llvm.ptr
    %84 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %85 = llvm.insertvalue %83, %84[0] : !llvm.struct<(ptr, ptr, i64)> 
    %86 = llvm.insertvalue %83, %85[1] : !llvm.struct<(ptr, ptr, i64)> 
    %87 = llvm.mlir.constant(0 : index) : i64
    %88 = llvm.insertvalue %87, %86[2] : !llvm.struct<(ptr, ptr, i64)> 
    %89 = builtin.unrealized_conversion_cast %88 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    %90 = llvm.extractvalue %88[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %0, %90 : f32, !llvm.ptr
    call @deriche_kernel_0(%71, %62, %89, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%17, %8, %53, %44, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%80, %71, %62, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%35, %26, %17, %8, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}


// -----// IR Dump After ExpandOps (memref-expand) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.undef : f32
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.alloca %1 x f32 : (i64) -> !llvm.ptr
    %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<(ptr, ptr, i64)> 
    %5 = llvm.insertvalue %2, %4[1] : !llvm.struct<(ptr, ptr, i64)> 
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.insertvalue %6, %5[2] : !llvm.struct<(ptr, ptr, i64)> 
    %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %2 : f32, !llvm.ptr
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.alloca %9 x f32 : (i64) -> !llvm.ptr
    %11 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.struct<(ptr, ptr, i64)> 
    %13 = llvm.insertvalue %10, %12[1] : !llvm.struct<(ptr, ptr, i64)> 
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.insertvalue %14, %13[2] : !llvm.struct<(ptr, ptr, i64)> 
    %16 = builtin.unrealized_conversion_cast %15 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %10 : f32, !llvm.ptr
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.alloca %17 x f32 : (i64) -> !llvm.ptr
    %19 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(ptr, ptr, i64)> 
    %21 = llvm.insertvalue %18, %20[1] : !llvm.struct<(ptr, ptr, i64)> 
    %22 = llvm.mlir.constant(0 : index) : i64
    %23 = llvm.insertvalue %22, %21[2] : !llvm.struct<(ptr, ptr, i64)> 
    %24 = builtin.unrealized_conversion_cast %23 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %18 : f32, !llvm.ptr
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.alloca %25 x f32 : (i64) -> !llvm.ptr
    %27 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %28 = llvm.insertvalue %26, %27[0] : !llvm.struct<(ptr, ptr, i64)> 
    %29 = llvm.insertvalue %26, %28[1] : !llvm.struct<(ptr, ptr, i64)> 
    %30 = llvm.mlir.constant(0 : index) : i64
    %31 = llvm.insertvalue %30, %29[2] : !llvm.struct<(ptr, ptr, i64)> 
    %32 = builtin.unrealized_conversion_cast %31 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %26 : f32, !llvm.ptr
    %33 = llvm.mlir.constant(1 : index) : i64
    %34 = llvm.alloca %33 x f32 : (i64) -> !llvm.ptr
    %35 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %36 = llvm.insertvalue %34, %35[0] : !llvm.struct<(ptr, ptr, i64)> 
    %37 = llvm.insertvalue %34, %36[1] : !llvm.struct<(ptr, ptr, i64)> 
    %38 = llvm.mlir.constant(0 : index) : i64
    %39 = llvm.insertvalue %38, %37[2] : !llvm.struct<(ptr, ptr, i64)> 
    %40 = builtin.unrealized_conversion_cast %39 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %34 : f32, !llvm.ptr
    %41 = llvm.mlir.constant(1 : index) : i64
    %42 = llvm.alloca %41 x f32 : (i64) -> !llvm.ptr
    %43 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %44 = llvm.insertvalue %42, %43[0] : !llvm.struct<(ptr, ptr, i64)> 
    %45 = llvm.insertvalue %42, %44[1] : !llvm.struct<(ptr, ptr, i64)> 
    %46 = llvm.mlir.constant(0 : index) : i64
    %47 = llvm.insertvalue %46, %45[2] : !llvm.struct<(ptr, ptr, i64)> 
    %48 = builtin.unrealized_conversion_cast %47 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %42 : f32, !llvm.ptr
    %49 = llvm.mlir.constant(1 : index) : i64
    %50 = llvm.alloca %49 x f32 : (i64) -> !llvm.ptr
    %51 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %52 = llvm.insertvalue %50, %51[0] : !llvm.struct<(ptr, ptr, i64)> 
    %53 = llvm.insertvalue %50, %52[1] : !llvm.struct<(ptr, ptr, i64)> 
    %54 = llvm.mlir.constant(0 : index) : i64
    %55 = llvm.insertvalue %54, %53[2] : !llvm.struct<(ptr, ptr, i64)> 
    %56 = builtin.unrealized_conversion_cast %55 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %50 : f32, !llvm.ptr
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.alloca %57 x f32 : (i64) -> !llvm.ptr
    %59 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(ptr, ptr, i64)> 
    %61 = llvm.insertvalue %58, %60[1] : !llvm.struct<(ptr, ptr, i64)> 
    %62 = llvm.mlir.constant(0 : index) : i64
    %63 = llvm.insertvalue %62, %61[2] : !llvm.struct<(ptr, ptr, i64)> 
    %64 = builtin.unrealized_conversion_cast %63 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %58 : f32, !llvm.ptr
    %65 = llvm.mlir.constant(1 : index) : i64
    %66 = llvm.alloca %65 x f32 : (i64) -> !llvm.ptr
    %67 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %68 = llvm.insertvalue %66, %67[0] : !llvm.struct<(ptr, ptr, i64)> 
    %69 = llvm.insertvalue %66, %68[1] : !llvm.struct<(ptr, ptr, i64)> 
    %70 = llvm.mlir.constant(0 : index) : i64
    %71 = llvm.insertvalue %70, %69[2] : !llvm.struct<(ptr, ptr, i64)> 
    %72 = builtin.unrealized_conversion_cast %71 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %66 : f32, !llvm.ptr
    %73 = llvm.mlir.constant(1 : index) : i64
    %74 = llvm.alloca %73 x f32 : (i64) -> !llvm.ptr
    %75 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %76 = llvm.insertvalue %74, %75[0] : !llvm.struct<(ptr, ptr, i64)> 
    %77 = llvm.insertvalue %74, %76[1] : !llvm.struct<(ptr, ptr, i64)> 
    %78 = llvm.mlir.constant(0 : index) : i64
    %79 = llvm.insertvalue %78, %77[2] : !llvm.struct<(ptr, ptr, i64)> 
    %80 = builtin.unrealized_conversion_cast %79 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %74 : f32, !llvm.ptr
    call @deriche_kernel_0(%64, %56, %80, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%16, %8, %48, %40, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%72, %64, %56, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%32, %24, %16, %8, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}


// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.undef : f32
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.alloca %1 x f32 : (i64) -> !llvm.ptr
    %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<(ptr, ptr, i64)> 
    %5 = llvm.insertvalue %2, %4[1] : !llvm.struct<(ptr, ptr, i64)> 
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.insertvalue %6, %5[2] : !llvm.struct<(ptr, ptr, i64)> 
    %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %2 : f32, !llvm.ptr
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.alloca %9 x f32 : (i64) -> !llvm.ptr
    %11 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.struct<(ptr, ptr, i64)> 
    %13 = llvm.insertvalue %10, %12[1] : !llvm.struct<(ptr, ptr, i64)> 
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.insertvalue %14, %13[2] : !llvm.struct<(ptr, ptr, i64)> 
    %16 = builtin.unrealized_conversion_cast %15 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %10 : f32, !llvm.ptr
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.alloca %17 x f32 : (i64) -> !llvm.ptr
    %19 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(ptr, ptr, i64)> 
    %21 = llvm.insertvalue %18, %20[1] : !llvm.struct<(ptr, ptr, i64)> 
    %22 = llvm.mlir.constant(0 : index) : i64
    %23 = llvm.insertvalue %22, %21[2] : !llvm.struct<(ptr, ptr, i64)> 
    %24 = builtin.unrealized_conversion_cast %23 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %18 : f32, !llvm.ptr
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.alloca %25 x f32 : (i64) -> !llvm.ptr
    %27 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %28 = llvm.insertvalue %26, %27[0] : !llvm.struct<(ptr, ptr, i64)> 
    %29 = llvm.insertvalue %26, %28[1] : !llvm.struct<(ptr, ptr, i64)> 
    %30 = llvm.mlir.constant(0 : index) : i64
    %31 = llvm.insertvalue %30, %29[2] : !llvm.struct<(ptr, ptr, i64)> 
    %32 = builtin.unrealized_conversion_cast %31 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %26 : f32, !llvm.ptr
    %33 = llvm.mlir.constant(1 : index) : i64
    %34 = llvm.alloca %33 x f32 : (i64) -> !llvm.ptr
    %35 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %36 = llvm.insertvalue %34, %35[0] : !llvm.struct<(ptr, ptr, i64)> 
    %37 = llvm.insertvalue %34, %36[1] : !llvm.struct<(ptr, ptr, i64)> 
    %38 = llvm.mlir.constant(0 : index) : i64
    %39 = llvm.insertvalue %38, %37[2] : !llvm.struct<(ptr, ptr, i64)> 
    %40 = builtin.unrealized_conversion_cast %39 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %34 : f32, !llvm.ptr
    %41 = llvm.mlir.constant(1 : index) : i64
    %42 = llvm.alloca %41 x f32 : (i64) -> !llvm.ptr
    %43 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %44 = llvm.insertvalue %42, %43[0] : !llvm.struct<(ptr, ptr, i64)> 
    %45 = llvm.insertvalue %42, %44[1] : !llvm.struct<(ptr, ptr, i64)> 
    %46 = llvm.mlir.constant(0 : index) : i64
    %47 = llvm.insertvalue %46, %45[2] : !llvm.struct<(ptr, ptr, i64)> 
    %48 = builtin.unrealized_conversion_cast %47 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %42 : f32, !llvm.ptr
    %49 = llvm.mlir.constant(1 : index) : i64
    %50 = llvm.alloca %49 x f32 : (i64) -> !llvm.ptr
    %51 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %52 = llvm.insertvalue %50, %51[0] : !llvm.struct<(ptr, ptr, i64)> 
    %53 = llvm.insertvalue %50, %52[1] : !llvm.struct<(ptr, ptr, i64)> 
    %54 = llvm.mlir.constant(0 : index) : i64
    %55 = llvm.insertvalue %54, %53[2] : !llvm.struct<(ptr, ptr, i64)> 
    %56 = builtin.unrealized_conversion_cast %55 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %50 : f32, !llvm.ptr
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.alloca %57 x f32 : (i64) -> !llvm.ptr
    %59 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(ptr, ptr, i64)> 
    %61 = llvm.insertvalue %58, %60[1] : !llvm.struct<(ptr, ptr, i64)> 
    %62 = llvm.mlir.constant(0 : index) : i64
    %63 = llvm.insertvalue %62, %61[2] : !llvm.struct<(ptr, ptr, i64)> 
    %64 = builtin.unrealized_conversion_cast %63 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %58 : f32, !llvm.ptr
    %65 = llvm.mlir.constant(1 : index) : i64
    %66 = llvm.alloca %65 x f32 : (i64) -> !llvm.ptr
    %67 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %68 = llvm.insertvalue %66, %67[0] : !llvm.struct<(ptr, ptr, i64)> 
    %69 = llvm.insertvalue %66, %68[1] : !llvm.struct<(ptr, ptr, i64)> 
    %70 = llvm.mlir.constant(0 : index) : i64
    %71 = llvm.insertvalue %70, %69[2] : !llvm.struct<(ptr, ptr, i64)> 
    %72 = builtin.unrealized_conversion_cast %71 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %66 : f32, !llvm.ptr
    %73 = llvm.mlir.constant(1 : index) : i64
    %74 = llvm.alloca %73 x f32 : (i64) -> !llvm.ptr
    %75 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %76 = llvm.insertvalue %74, %75[0] : !llvm.struct<(ptr, ptr, i64)> 
    %77 = llvm.insertvalue %74, %76[1] : !llvm.struct<(ptr, ptr, i64)> 
    %78 = llvm.mlir.constant(0 : index) : i64
    %79 = llvm.insertvalue %78, %77[2] : !llvm.struct<(ptr, ptr, i64)> 
    %80 = builtin.unrealized_conversion_cast %79 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %0, %74 : f32, !llvm.ptr
    call @deriche_kernel_0(%64, %56, %80, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%16, %8, %48, %40, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%72, %64, %56, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%32, %24, %16, %8, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}


// -----// IR Dump After ConvertFuncToLLVMPass (convert-func-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @deriche(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.mlir.constant(4096 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.mlir.constant(2160 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.mlir.constant(2160 : index) : i64
    %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg1, %13[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg1, %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(4096 : index) : i64
    %19 = llvm.insertvalue %18, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.mlir.constant(2160 : index) : i64
    %21 = llvm.insertvalue %20, %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.mlir.constant(2160 : index) : i64
    %23 = llvm.insertvalue %22, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.insertvalue %24, %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.insertvalue %arg2, %26[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %arg2, %27[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.mlir.constant(0 : index) : i64
    %30 = llvm.insertvalue %29, %28[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.mlir.constant(4096 : index) : i64
    %32 = llvm.insertvalue %31, %30[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.constant(2160 : index) : i64
    %34 = llvm.insertvalue %33, %32[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.mlir.constant(2160 : index) : i64
    %36 = llvm.insertvalue %35, %34[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.mlir.constant(1 : index) : i64
    %38 = llvm.insertvalue %37, %36[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %40 = llvm.insertvalue %arg3, %39[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.insertvalue %arg3, %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.mlir.constant(0 : index) : i64
    %43 = llvm.insertvalue %42, %41[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.mlir.constant(4096 : index) : i64
    %45 = llvm.insertvalue %44, %43[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.mlir.constant(2160 : index) : i64
    %47 = llvm.insertvalue %46, %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.mlir.constant(2160 : index) : i64
    %49 = llvm.insertvalue %48, %47[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.mlir.constant(1 : index) : i64
    %51 = llvm.insertvalue %50, %49[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.mlir.undef : f32
    %53 = llvm.mlir.constant(1 : index) : i64
    %54 = llvm.alloca %53 x f32 : (i64) -> !llvm.ptr
    %55 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %56 = llvm.insertvalue %54, %55[0] : !llvm.struct<(ptr, ptr, i64)> 
    %57 = llvm.insertvalue %54, %56[1] : !llvm.struct<(ptr, ptr, i64)> 
    %58 = llvm.mlir.constant(0 : index) : i64
    %59 = llvm.insertvalue %58, %57[2] : !llvm.struct<(ptr, ptr, i64)> 
    %60 = builtin.unrealized_conversion_cast %59 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %54 : f32, !llvm.ptr
    %61 = llvm.mlir.constant(1 : index) : i64
    %62 = llvm.alloca %61 x f32 : (i64) -> !llvm.ptr
    %63 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %64 = llvm.insertvalue %62, %63[0] : !llvm.struct<(ptr, ptr, i64)> 
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr, ptr, i64)> 
    %66 = llvm.mlir.constant(0 : index) : i64
    %67 = llvm.insertvalue %66, %65[2] : !llvm.struct<(ptr, ptr, i64)> 
    %68 = builtin.unrealized_conversion_cast %67 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %62 : f32, !llvm.ptr
    %69 = llvm.mlir.constant(1 : index) : i64
    %70 = llvm.alloca %69 x f32 : (i64) -> !llvm.ptr
    %71 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %72 = llvm.insertvalue %70, %71[0] : !llvm.struct<(ptr, ptr, i64)> 
    %73 = llvm.insertvalue %70, %72[1] : !llvm.struct<(ptr, ptr, i64)> 
    %74 = llvm.mlir.constant(0 : index) : i64
    %75 = llvm.insertvalue %74, %73[2] : !llvm.struct<(ptr, ptr, i64)> 
    %76 = builtin.unrealized_conversion_cast %75 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %70 : f32, !llvm.ptr
    %77 = llvm.mlir.constant(1 : index) : i64
    %78 = llvm.alloca %77 x f32 : (i64) -> !llvm.ptr
    %79 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %80 = llvm.insertvalue %78, %79[0] : !llvm.struct<(ptr, ptr, i64)> 
    %81 = llvm.insertvalue %78, %80[1] : !llvm.struct<(ptr, ptr, i64)> 
    %82 = llvm.mlir.constant(0 : index) : i64
    %83 = llvm.insertvalue %82, %81[2] : !llvm.struct<(ptr, ptr, i64)> 
    %84 = builtin.unrealized_conversion_cast %83 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %78 : f32, !llvm.ptr
    %85 = llvm.mlir.constant(1 : index) : i64
    %86 = llvm.alloca %85 x f32 : (i64) -> !llvm.ptr
    %87 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %88 = llvm.insertvalue %86, %87[0] : !llvm.struct<(ptr, ptr, i64)> 
    %89 = llvm.insertvalue %86, %88[1] : !llvm.struct<(ptr, ptr, i64)> 
    %90 = llvm.mlir.constant(0 : index) : i64
    %91 = llvm.insertvalue %90, %89[2] : !llvm.struct<(ptr, ptr, i64)> 
    %92 = builtin.unrealized_conversion_cast %91 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %86 : f32, !llvm.ptr
    %93 = llvm.mlir.constant(1 : index) : i64
    %94 = llvm.alloca %93 x f32 : (i64) -> !llvm.ptr
    %95 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %96 = llvm.insertvalue %94, %95[0] : !llvm.struct<(ptr, ptr, i64)> 
    %97 = llvm.insertvalue %94, %96[1] : !llvm.struct<(ptr, ptr, i64)> 
    %98 = llvm.mlir.constant(0 : index) : i64
    %99 = llvm.insertvalue %98, %97[2] : !llvm.struct<(ptr, ptr, i64)> 
    %100 = builtin.unrealized_conversion_cast %99 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %94 : f32, !llvm.ptr
    %101 = llvm.mlir.constant(1 : index) : i64
    %102 = llvm.alloca %101 x f32 : (i64) -> !llvm.ptr
    %103 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %104 = llvm.insertvalue %102, %103[0] : !llvm.struct<(ptr, ptr, i64)> 
    %105 = llvm.insertvalue %102, %104[1] : !llvm.struct<(ptr, ptr, i64)> 
    %106 = llvm.mlir.constant(0 : index) : i64
    %107 = llvm.insertvalue %106, %105[2] : !llvm.struct<(ptr, ptr, i64)> 
    %108 = builtin.unrealized_conversion_cast %107 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %102 : f32, !llvm.ptr
    %109 = llvm.mlir.constant(1 : index) : i64
    %110 = llvm.alloca %109 x f32 : (i64) -> !llvm.ptr
    %111 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %112 = llvm.insertvalue %110, %111[0] : !llvm.struct<(ptr, ptr, i64)> 
    %113 = llvm.insertvalue %110, %112[1] : !llvm.struct<(ptr, ptr, i64)> 
    %114 = llvm.mlir.constant(0 : index) : i64
    %115 = llvm.insertvalue %114, %113[2] : !llvm.struct<(ptr, ptr, i64)> 
    %116 = builtin.unrealized_conversion_cast %115 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %110 : f32, !llvm.ptr
    %117 = llvm.mlir.constant(1 : index) : i64
    %118 = llvm.alloca %117 x f32 : (i64) -> !llvm.ptr
    %119 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %120 = llvm.insertvalue %118, %119[0] : !llvm.struct<(ptr, ptr, i64)> 
    %121 = llvm.insertvalue %118, %120[1] : !llvm.struct<(ptr, ptr, i64)> 
    %122 = llvm.mlir.constant(0 : index) : i64
    %123 = llvm.insertvalue %122, %121[2] : !llvm.struct<(ptr, ptr, i64)> 
    %124 = builtin.unrealized_conversion_cast %123 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %118 : f32, !llvm.ptr
    %125 = llvm.mlir.constant(1 : index) : i64
    %126 = llvm.alloca %125 x f32 : (i64) -> !llvm.ptr
    %127 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %128 = llvm.insertvalue %126, %127[0] : !llvm.struct<(ptr, ptr, i64)> 
    %129 = llvm.insertvalue %126, %128[1] : !llvm.struct<(ptr, ptr, i64)> 
    %130 = llvm.mlir.constant(0 : index) : i64
    %131 = llvm.insertvalue %130, %129[2] : !llvm.struct<(ptr, ptr, i64)> 
    %132 = builtin.unrealized_conversion_cast %131 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %126 : f32, !llvm.ptr
    %133 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64)> 
    %134 = llvm.extractvalue %107[1] : !llvm.struct<(ptr, ptr, i64)> 
    %135 = llvm.extractvalue %131[1] : !llvm.struct<(ptr, ptr, i64)> 
    %136 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_0(%133, %134, %135, %136, %137) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %138 = llvm.extractvalue %67[1] : !llvm.struct<(ptr, ptr, i64)> 
    %139 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64)> 
    %140 = llvm.extractvalue %99[1] : !llvm.struct<(ptr, ptr, i64)> 
    %141 = llvm.extractvalue %91[1] : !llvm.struct<(ptr, ptr, i64)> 
    %142 = llvm.extractvalue %51[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %143 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_1(%138, %139, %140, %141, %142, %143) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %144 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.extractvalue %51[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_2(%144, %145, %146) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %147 = llvm.extractvalue %123[1] : !llvm.struct<(ptr, ptr, i64)> 
    %148 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64)> 
    %149 = llvm.extractvalue %107[1] : !llvm.struct<(ptr, ptr, i64)> 
    %150 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_3(%147, %148, %149, %150, %151) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %152 = llvm.extractvalue %83[1] : !llvm.struct<(ptr, ptr, i64)> 
    %153 = llvm.extractvalue %75[1] : !llvm.struct<(ptr, ptr, i64)> 
    %154 = llvm.extractvalue %67[1] : !llvm.struct<(ptr, ptr, i64)> 
    %155 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64)> 
    %156 = llvm.extractvalue %51[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %157 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_4(%152, %153, %154, %155, %156, %157) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %158 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.extractvalue %51[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_5(%158, %159, %160) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @deriche_kernel_0(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_1(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_2(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_3(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_4(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_5(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}


// -----// IR Dump After ConvertFuncToLLVMPass (convert-func-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @deriche(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.mlir.constant(4096 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.mlir.constant(2160 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.mlir.constant(2160 : index) : i64
    %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg1, %13[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg1, %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(4096 : index) : i64
    %19 = llvm.insertvalue %18, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.mlir.constant(2160 : index) : i64
    %21 = llvm.insertvalue %20, %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.mlir.constant(2160 : index) : i64
    %23 = llvm.insertvalue %22, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.insertvalue %24, %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.insertvalue %arg2, %26[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %arg2, %27[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.mlir.constant(0 : index) : i64
    %30 = llvm.insertvalue %29, %28[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.mlir.constant(4096 : index) : i64
    %32 = llvm.insertvalue %31, %30[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.constant(2160 : index) : i64
    %34 = llvm.insertvalue %33, %32[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.mlir.constant(2160 : index) : i64
    %36 = llvm.insertvalue %35, %34[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.mlir.constant(1 : index) : i64
    %38 = llvm.insertvalue %37, %36[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %40 = llvm.insertvalue %arg3, %39[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.insertvalue %arg3, %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.mlir.constant(0 : index) : i64
    %43 = llvm.insertvalue %42, %41[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.mlir.constant(4096 : index) : i64
    %45 = llvm.insertvalue %44, %43[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.mlir.constant(2160 : index) : i64
    %47 = llvm.insertvalue %46, %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.mlir.constant(2160 : index) : i64
    %49 = llvm.insertvalue %48, %47[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.mlir.constant(1 : index) : i64
    %51 = llvm.insertvalue %50, %49[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.mlir.undef : f32
    %53 = llvm.mlir.constant(1 : index) : i64
    %54 = llvm.alloca %53 x f32 : (i64) -> !llvm.ptr
    %55 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %56 = llvm.insertvalue %54, %55[0] : !llvm.struct<(ptr, ptr, i64)> 
    %57 = llvm.insertvalue %54, %56[1] : !llvm.struct<(ptr, ptr, i64)> 
    %58 = llvm.mlir.constant(0 : index) : i64
    %59 = llvm.insertvalue %58, %57[2] : !llvm.struct<(ptr, ptr, i64)> 
    %60 = builtin.unrealized_conversion_cast %59 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %54 : f32, !llvm.ptr
    %61 = llvm.mlir.constant(1 : index) : i64
    %62 = llvm.alloca %61 x f32 : (i64) -> !llvm.ptr
    %63 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %64 = llvm.insertvalue %62, %63[0] : !llvm.struct<(ptr, ptr, i64)> 
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr, ptr, i64)> 
    %66 = llvm.mlir.constant(0 : index) : i64
    %67 = llvm.insertvalue %66, %65[2] : !llvm.struct<(ptr, ptr, i64)> 
    %68 = builtin.unrealized_conversion_cast %67 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %62 : f32, !llvm.ptr
    %69 = llvm.mlir.constant(1 : index) : i64
    %70 = llvm.alloca %69 x f32 : (i64) -> !llvm.ptr
    %71 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %72 = llvm.insertvalue %70, %71[0] : !llvm.struct<(ptr, ptr, i64)> 
    %73 = llvm.insertvalue %70, %72[1] : !llvm.struct<(ptr, ptr, i64)> 
    %74 = llvm.mlir.constant(0 : index) : i64
    %75 = llvm.insertvalue %74, %73[2] : !llvm.struct<(ptr, ptr, i64)> 
    %76 = builtin.unrealized_conversion_cast %75 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %70 : f32, !llvm.ptr
    %77 = llvm.mlir.constant(1 : index) : i64
    %78 = llvm.alloca %77 x f32 : (i64) -> !llvm.ptr
    %79 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %80 = llvm.insertvalue %78, %79[0] : !llvm.struct<(ptr, ptr, i64)> 
    %81 = llvm.insertvalue %78, %80[1] : !llvm.struct<(ptr, ptr, i64)> 
    %82 = llvm.mlir.constant(0 : index) : i64
    %83 = llvm.insertvalue %82, %81[2] : !llvm.struct<(ptr, ptr, i64)> 
    %84 = builtin.unrealized_conversion_cast %83 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %78 : f32, !llvm.ptr
    %85 = llvm.mlir.constant(1 : index) : i64
    %86 = llvm.alloca %85 x f32 : (i64) -> !llvm.ptr
    %87 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %88 = llvm.insertvalue %86, %87[0] : !llvm.struct<(ptr, ptr, i64)> 
    %89 = llvm.insertvalue %86, %88[1] : !llvm.struct<(ptr, ptr, i64)> 
    %90 = llvm.mlir.constant(0 : index) : i64
    %91 = llvm.insertvalue %90, %89[2] : !llvm.struct<(ptr, ptr, i64)> 
    %92 = builtin.unrealized_conversion_cast %91 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %86 : f32, !llvm.ptr
    %93 = llvm.mlir.constant(1 : index) : i64
    %94 = llvm.alloca %93 x f32 : (i64) -> !llvm.ptr
    %95 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %96 = llvm.insertvalue %94, %95[0] : !llvm.struct<(ptr, ptr, i64)> 
    %97 = llvm.insertvalue %94, %96[1] : !llvm.struct<(ptr, ptr, i64)> 
    %98 = llvm.mlir.constant(0 : index) : i64
    %99 = llvm.insertvalue %98, %97[2] : !llvm.struct<(ptr, ptr, i64)> 
    %100 = builtin.unrealized_conversion_cast %99 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %94 : f32, !llvm.ptr
    %101 = llvm.mlir.constant(1 : index) : i64
    %102 = llvm.alloca %101 x f32 : (i64) -> !llvm.ptr
    %103 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %104 = llvm.insertvalue %102, %103[0] : !llvm.struct<(ptr, ptr, i64)> 
    %105 = llvm.insertvalue %102, %104[1] : !llvm.struct<(ptr, ptr, i64)> 
    %106 = llvm.mlir.constant(0 : index) : i64
    %107 = llvm.insertvalue %106, %105[2] : !llvm.struct<(ptr, ptr, i64)> 
    %108 = builtin.unrealized_conversion_cast %107 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %102 : f32, !llvm.ptr
    %109 = llvm.mlir.constant(1 : index) : i64
    %110 = llvm.alloca %109 x f32 : (i64) -> !llvm.ptr
    %111 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %112 = llvm.insertvalue %110, %111[0] : !llvm.struct<(ptr, ptr, i64)> 
    %113 = llvm.insertvalue %110, %112[1] : !llvm.struct<(ptr, ptr, i64)> 
    %114 = llvm.mlir.constant(0 : index) : i64
    %115 = llvm.insertvalue %114, %113[2] : !llvm.struct<(ptr, ptr, i64)> 
    %116 = builtin.unrealized_conversion_cast %115 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %110 : f32, !llvm.ptr
    %117 = llvm.mlir.constant(1 : index) : i64
    %118 = llvm.alloca %117 x f32 : (i64) -> !llvm.ptr
    %119 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %120 = llvm.insertvalue %118, %119[0] : !llvm.struct<(ptr, ptr, i64)> 
    %121 = llvm.insertvalue %118, %120[1] : !llvm.struct<(ptr, ptr, i64)> 
    %122 = llvm.mlir.constant(0 : index) : i64
    %123 = llvm.insertvalue %122, %121[2] : !llvm.struct<(ptr, ptr, i64)> 
    %124 = builtin.unrealized_conversion_cast %123 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %118 : f32, !llvm.ptr
    %125 = llvm.mlir.constant(1 : index) : i64
    %126 = llvm.alloca %125 x f32 : (i64) -> !llvm.ptr
    %127 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %128 = llvm.insertvalue %126, %127[0] : !llvm.struct<(ptr, ptr, i64)> 
    %129 = llvm.insertvalue %126, %128[1] : !llvm.struct<(ptr, ptr, i64)> 
    %130 = llvm.mlir.constant(0 : index) : i64
    %131 = llvm.insertvalue %130, %129[2] : !llvm.struct<(ptr, ptr, i64)> 
    %132 = builtin.unrealized_conversion_cast %131 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %126 : f32, !llvm.ptr
    %133 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64)> 
    %134 = llvm.extractvalue %107[1] : !llvm.struct<(ptr, ptr, i64)> 
    %135 = llvm.extractvalue %131[1] : !llvm.struct<(ptr, ptr, i64)> 
    %136 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_0(%133, %134, %135, %136, %137) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %138 = llvm.extractvalue %67[1] : !llvm.struct<(ptr, ptr, i64)> 
    %139 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64)> 
    %140 = llvm.extractvalue %99[1] : !llvm.struct<(ptr, ptr, i64)> 
    %141 = llvm.extractvalue %91[1] : !llvm.struct<(ptr, ptr, i64)> 
    %142 = llvm.extractvalue %51[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %143 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_1(%138, %139, %140, %141, %142, %143) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %144 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.extractvalue %51[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_2(%144, %145, %146) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %147 = llvm.extractvalue %123[1] : !llvm.struct<(ptr, ptr, i64)> 
    %148 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64)> 
    %149 = llvm.extractvalue %107[1] : !llvm.struct<(ptr, ptr, i64)> 
    %150 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_3(%147, %148, %149, %150, %151) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %152 = llvm.extractvalue %83[1] : !llvm.struct<(ptr, ptr, i64)> 
    %153 = llvm.extractvalue %75[1] : !llvm.struct<(ptr, ptr, i64)> 
    %154 = llvm.extractvalue %67[1] : !llvm.struct<(ptr, ptr, i64)> 
    %155 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64)> 
    %156 = llvm.extractvalue %51[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %157 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_4(%152, %153, %154, %155, %156, %157) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %158 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.extractvalue %51[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_5(%158, %159, %160) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @deriche_kernel_0(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_1(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_2(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_3(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_4(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_5(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}


// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @deriche(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.mlir.constant(4096 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.mlir.constant(2160 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.mlir.constant(2160 : index) : i64
    %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg1, %13[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg1, %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(4096 : index) : i64
    %19 = llvm.insertvalue %18, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.mlir.constant(2160 : index) : i64
    %21 = llvm.insertvalue %20, %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.mlir.constant(2160 : index) : i64
    %23 = llvm.insertvalue %22, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.insertvalue %24, %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.insertvalue %arg2, %26[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %arg2, %27[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.mlir.constant(0 : index) : i64
    %30 = llvm.insertvalue %29, %28[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.mlir.constant(4096 : index) : i64
    %32 = llvm.insertvalue %31, %30[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.constant(2160 : index) : i64
    %34 = llvm.insertvalue %33, %32[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.mlir.constant(2160 : index) : i64
    %36 = llvm.insertvalue %35, %34[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.mlir.constant(1 : index) : i64
    %38 = llvm.insertvalue %37, %36[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %40 = llvm.insertvalue %arg3, %39[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.insertvalue %arg3, %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.mlir.constant(0 : index) : i64
    %43 = llvm.insertvalue %42, %41[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.mlir.constant(4096 : index) : i64
    %45 = llvm.insertvalue %44, %43[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.mlir.constant(2160 : index) : i64
    %47 = llvm.insertvalue %46, %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.mlir.constant(2160 : index) : i64
    %49 = llvm.insertvalue %48, %47[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.mlir.constant(1 : index) : i64
    %51 = llvm.insertvalue %50, %49[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.mlir.undef : f32
    %53 = llvm.mlir.constant(1 : index) : i64
    %54 = llvm.alloca %53 x f32 : (i64) -> !llvm.ptr
    %55 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %56 = llvm.insertvalue %54, %55[0] : !llvm.struct<(ptr, ptr, i64)> 
    %57 = llvm.insertvalue %54, %56[1] : !llvm.struct<(ptr, ptr, i64)> 
    %58 = llvm.mlir.constant(0 : index) : i64
    %59 = llvm.insertvalue %58, %57[2] : !llvm.struct<(ptr, ptr, i64)> 
    %60 = builtin.unrealized_conversion_cast %59 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %54 : f32, !llvm.ptr
    %61 = llvm.mlir.constant(1 : index) : i64
    %62 = llvm.alloca %61 x f32 : (i64) -> !llvm.ptr
    %63 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %64 = llvm.insertvalue %62, %63[0] : !llvm.struct<(ptr, ptr, i64)> 
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr, ptr, i64)> 
    %66 = llvm.mlir.constant(0 : index) : i64
    %67 = llvm.insertvalue %66, %65[2] : !llvm.struct<(ptr, ptr, i64)> 
    %68 = builtin.unrealized_conversion_cast %67 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %62 : f32, !llvm.ptr
    %69 = llvm.mlir.constant(1 : index) : i64
    %70 = llvm.alloca %69 x f32 : (i64) -> !llvm.ptr
    %71 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %72 = llvm.insertvalue %70, %71[0] : !llvm.struct<(ptr, ptr, i64)> 
    %73 = llvm.insertvalue %70, %72[1] : !llvm.struct<(ptr, ptr, i64)> 
    %74 = llvm.mlir.constant(0 : index) : i64
    %75 = llvm.insertvalue %74, %73[2] : !llvm.struct<(ptr, ptr, i64)> 
    %76 = builtin.unrealized_conversion_cast %75 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %70 : f32, !llvm.ptr
    %77 = llvm.mlir.constant(1 : index) : i64
    %78 = llvm.alloca %77 x f32 : (i64) -> !llvm.ptr
    %79 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %80 = llvm.insertvalue %78, %79[0] : !llvm.struct<(ptr, ptr, i64)> 
    %81 = llvm.insertvalue %78, %80[1] : !llvm.struct<(ptr, ptr, i64)> 
    %82 = llvm.mlir.constant(0 : index) : i64
    %83 = llvm.insertvalue %82, %81[2] : !llvm.struct<(ptr, ptr, i64)> 
    %84 = builtin.unrealized_conversion_cast %83 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %78 : f32, !llvm.ptr
    %85 = llvm.mlir.constant(1 : index) : i64
    %86 = llvm.alloca %85 x f32 : (i64) -> !llvm.ptr
    %87 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %88 = llvm.insertvalue %86, %87[0] : !llvm.struct<(ptr, ptr, i64)> 
    %89 = llvm.insertvalue %86, %88[1] : !llvm.struct<(ptr, ptr, i64)> 
    %90 = llvm.mlir.constant(0 : index) : i64
    %91 = llvm.insertvalue %90, %89[2] : !llvm.struct<(ptr, ptr, i64)> 
    %92 = builtin.unrealized_conversion_cast %91 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %86 : f32, !llvm.ptr
    %93 = llvm.mlir.constant(1 : index) : i64
    %94 = llvm.alloca %93 x f32 : (i64) -> !llvm.ptr
    %95 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %96 = llvm.insertvalue %94, %95[0] : !llvm.struct<(ptr, ptr, i64)> 
    %97 = llvm.insertvalue %94, %96[1] : !llvm.struct<(ptr, ptr, i64)> 
    %98 = llvm.mlir.constant(0 : index) : i64
    %99 = llvm.insertvalue %98, %97[2] : !llvm.struct<(ptr, ptr, i64)> 
    %100 = builtin.unrealized_conversion_cast %99 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %94 : f32, !llvm.ptr
    %101 = llvm.mlir.constant(1 : index) : i64
    %102 = llvm.alloca %101 x f32 : (i64) -> !llvm.ptr
    %103 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %104 = llvm.insertvalue %102, %103[0] : !llvm.struct<(ptr, ptr, i64)> 
    %105 = llvm.insertvalue %102, %104[1] : !llvm.struct<(ptr, ptr, i64)> 
    %106 = llvm.mlir.constant(0 : index) : i64
    %107 = llvm.insertvalue %106, %105[2] : !llvm.struct<(ptr, ptr, i64)> 
    %108 = builtin.unrealized_conversion_cast %107 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %102 : f32, !llvm.ptr
    %109 = llvm.mlir.constant(1 : index) : i64
    %110 = llvm.alloca %109 x f32 : (i64) -> !llvm.ptr
    %111 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %112 = llvm.insertvalue %110, %111[0] : !llvm.struct<(ptr, ptr, i64)> 
    %113 = llvm.insertvalue %110, %112[1] : !llvm.struct<(ptr, ptr, i64)> 
    %114 = llvm.mlir.constant(0 : index) : i64
    %115 = llvm.insertvalue %114, %113[2] : !llvm.struct<(ptr, ptr, i64)> 
    %116 = builtin.unrealized_conversion_cast %115 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %110 : f32, !llvm.ptr
    %117 = llvm.mlir.constant(1 : index) : i64
    %118 = llvm.alloca %117 x f32 : (i64) -> !llvm.ptr
    %119 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %120 = llvm.insertvalue %118, %119[0] : !llvm.struct<(ptr, ptr, i64)> 
    %121 = llvm.insertvalue %118, %120[1] : !llvm.struct<(ptr, ptr, i64)> 
    %122 = llvm.mlir.constant(0 : index) : i64
    %123 = llvm.insertvalue %122, %121[2] : !llvm.struct<(ptr, ptr, i64)> 
    %124 = builtin.unrealized_conversion_cast %123 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %118 : f32, !llvm.ptr
    %125 = llvm.mlir.constant(1 : index) : i64
    %126 = llvm.alloca %125 x f32 : (i64) -> !llvm.ptr
    %127 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %128 = llvm.insertvalue %126, %127[0] : !llvm.struct<(ptr, ptr, i64)> 
    %129 = llvm.insertvalue %126, %128[1] : !llvm.struct<(ptr, ptr, i64)> 
    %130 = llvm.mlir.constant(0 : index) : i64
    %131 = llvm.insertvalue %130, %129[2] : !llvm.struct<(ptr, ptr, i64)> 
    %132 = builtin.unrealized_conversion_cast %131 : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
    llvm.store %52, %126 : f32, !llvm.ptr
    %133 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64)> 
    %134 = llvm.extractvalue %107[1] : !llvm.struct<(ptr, ptr, i64)> 
    %135 = llvm.extractvalue %131[1] : !llvm.struct<(ptr, ptr, i64)> 
    %136 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_0(%133, %134, %135, %136, %137) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %138 = llvm.extractvalue %67[1] : !llvm.struct<(ptr, ptr, i64)> 
    %139 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64)> 
    %140 = llvm.extractvalue %99[1] : !llvm.struct<(ptr, ptr, i64)> 
    %141 = llvm.extractvalue %91[1] : !llvm.struct<(ptr, ptr, i64)> 
    %142 = llvm.extractvalue %51[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %143 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_1(%138, %139, %140, %141, %142, %143) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %144 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.extractvalue %51[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_2(%144, %145, %146) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %147 = llvm.extractvalue %123[1] : !llvm.struct<(ptr, ptr, i64)> 
    %148 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64)> 
    %149 = llvm.extractvalue %107[1] : !llvm.struct<(ptr, ptr, i64)> 
    %150 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_3(%147, %148, %149, %150, %151) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %152 = llvm.extractvalue %83[1] : !llvm.struct<(ptr, ptr, i64)> 
    %153 = llvm.extractvalue %75[1] : !llvm.struct<(ptr, ptr, i64)> 
    %154 = llvm.extractvalue %67[1] : !llvm.struct<(ptr, ptr, i64)> 
    %155 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64)> 
    %156 = llvm.extractvalue %51[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %157 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_4(%152, %153, %154, %155, %156, %157) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %158 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.extractvalue %51[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_5(%158, %159, %160) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @deriche_kernel_0(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_1(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_2(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_3(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_4(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_5(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}


// -----// IR Dump After CSE (cse) //----- //
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @deriche(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.mlir.constant(4096 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.mlir.constant(2160 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.insertvalue %7, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.insertvalue %10, %9[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg1, %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %3, %13[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %5, %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %7, %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %7, %16[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %10, %17[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %arg2, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %arg2, %19[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %3, %20[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %5, %21[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %7, %22[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.insertvalue %7, %23[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %10, %24[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %arg3, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %arg3, %26[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %3, %27[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %5, %28[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %7, %29[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %7, %30[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %10, %31[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.undef : f32
    %34 = llvm.alloca %10 x f32 : (i64) -> !llvm.ptr
    %35 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %36 = llvm.insertvalue %34, %35[0] : !llvm.struct<(ptr, ptr, i64)> 
    %37 = llvm.insertvalue %34, %36[1] : !llvm.struct<(ptr, ptr, i64)> 
    %38 = llvm.insertvalue %3, %37[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %33, %34 : f32, !llvm.ptr
    %39 = llvm.alloca %10 x f32 : (i64) -> !llvm.ptr
    %40 = llvm.insertvalue %39, %35[0] : !llvm.struct<(ptr, ptr, i64)> 
    %41 = llvm.insertvalue %39, %40[1] : !llvm.struct<(ptr, ptr, i64)> 
    %42 = llvm.insertvalue %3, %41[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %33, %39 : f32, !llvm.ptr
    %43 = llvm.alloca %10 x f32 : (i64) -> !llvm.ptr
    %44 = llvm.insertvalue %43, %35[0] : !llvm.struct<(ptr, ptr, i64)> 
    %45 = llvm.insertvalue %43, %44[1] : !llvm.struct<(ptr, ptr, i64)> 
    %46 = llvm.insertvalue %3, %45[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %33, %43 : f32, !llvm.ptr
    %47 = llvm.alloca %10 x f32 : (i64) -> !llvm.ptr
    %48 = llvm.insertvalue %47, %35[0] : !llvm.struct<(ptr, ptr, i64)> 
    %49 = llvm.insertvalue %47, %48[1] : !llvm.struct<(ptr, ptr, i64)> 
    %50 = llvm.insertvalue %3, %49[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %33, %47 : f32, !llvm.ptr
    %51 = llvm.alloca %10 x f32 : (i64) -> !llvm.ptr
    %52 = llvm.insertvalue %51, %35[0] : !llvm.struct<(ptr, ptr, i64)> 
    %53 = llvm.insertvalue %51, %52[1] : !llvm.struct<(ptr, ptr, i64)> 
    %54 = llvm.insertvalue %3, %53[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %33, %51 : f32, !llvm.ptr
    %55 = llvm.alloca %10 x f32 : (i64) -> !llvm.ptr
    %56 = llvm.insertvalue %55, %35[0] : !llvm.struct<(ptr, ptr, i64)> 
    %57 = llvm.insertvalue %55, %56[1] : !llvm.struct<(ptr, ptr, i64)> 
    %58 = llvm.insertvalue %3, %57[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %33, %55 : f32, !llvm.ptr
    %59 = llvm.alloca %10 x f32 : (i64) -> !llvm.ptr
    %60 = llvm.insertvalue %59, %35[0] : !llvm.struct<(ptr, ptr, i64)> 
    %61 = llvm.insertvalue %59, %60[1] : !llvm.struct<(ptr, ptr, i64)> 
    %62 = llvm.insertvalue %3, %61[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %33, %59 : f32, !llvm.ptr
    %63 = llvm.alloca %10 x f32 : (i64) -> !llvm.ptr
    %64 = llvm.insertvalue %63, %35[0] : !llvm.struct<(ptr, ptr, i64)> 
    %65 = llvm.insertvalue %63, %64[1] : !llvm.struct<(ptr, ptr, i64)> 
    %66 = llvm.insertvalue %3, %65[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %33, %63 : f32, !llvm.ptr
    %67 = llvm.alloca %10 x f32 : (i64) -> !llvm.ptr
    %68 = llvm.insertvalue %67, %35[0] : !llvm.struct<(ptr, ptr, i64)> 
    %69 = llvm.insertvalue %67, %68[1] : !llvm.struct<(ptr, ptr, i64)> 
    %70 = llvm.insertvalue %3, %69[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %33, %67 : f32, !llvm.ptr
    %71 = llvm.alloca %10 x f32 : (i64) -> !llvm.ptr
    %72 = llvm.insertvalue %71, %35[0] : !llvm.struct<(ptr, ptr, i64)> 
    %73 = llvm.insertvalue %71, %72[1] : !llvm.struct<(ptr, ptr, i64)> 
    %74 = llvm.insertvalue %3, %73[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %33, %71 : f32, !llvm.ptr
    %75 = llvm.extractvalue %66[1] : !llvm.struct<(ptr, ptr, i64)> 
    %76 = llvm.extractvalue %62[1] : !llvm.struct<(ptr, ptr, i64)> 
    %77 = llvm.extractvalue %74[1] : !llvm.struct<(ptr, ptr, i64)> 
    %78 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_0(%75, %76, %77, %78, %79) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %80 = llvm.extractvalue %42[1] : !llvm.struct<(ptr, ptr, i64)> 
    %81 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64)> 
    %82 = llvm.extractvalue %58[1] : !llvm.struct<(ptr, ptr, i64)> 
    %83 = llvm.extractvalue %54[1] : !llvm.struct<(ptr, ptr, i64)> 
    %84 = llvm.extractvalue %32[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_1(%80, %81, %82, %83, %84, %78) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %85 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @deriche_kernel_2(%79, %84, %85) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %86 = llvm.extractvalue %70[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.call @deriche_kernel_3(%86, %75, %76, %85, %79) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %87 = llvm.extractvalue %50[1] : !llvm.struct<(ptr, ptr, i64)> 
    %88 = llvm.extractvalue %46[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.call @deriche_kernel_4(%87, %88, %80, %81, %84, %85) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @deriche_kernel_5(%79, %84, %85) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @deriche_kernel_0(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_1(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_2(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_3(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_4(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @deriche_kernel_5(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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


// -----// IR Dump After ReconcileUnrealizedCasts (reconcile-unrealized-casts) //----- //
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


