// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)

// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
  call @forward_kernel_0(%arg0, %alloc) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
  return %alloc : memref<1x128x64xf32>
}

// -----// IR Dump After ArithExpandOps (arith-expand) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %alloc) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %alloc : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}


// -----// IR Dump After ExpandOps (memref-expand) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %alloc) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %alloc : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}


// -----// IR Dump After NormalizeMemRefs (normalize-memrefs) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %alloc) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %alloc : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}


// -----// IR Dump After ExpandStridedMetadata (expand-strided-metadata) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %alloc) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %alloc : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}


// -----// IR Dump After ConvertAffineToStandard (lower-affine) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %alloc) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %alloc : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}


// -----// IR Dump After SCFForLoopCanonicalization (scf-for-loop-canonicalization) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %alloc) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %alloc : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}


// -----// IR Dump After SCFToControlFlow (convert-scf-to-cf) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %alloc) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %alloc : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}


// -----// IR Dump After ConvertMathToLLVMPass (convert-math-to-llvm) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %alloc) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %alloc : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}


// -----// IR Dump After ConvertMathToLibm (convert-math-to-libm) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %alloc) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %alloc : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}


// -----// IR Dump After ArithToLLVMConversionPass (convert-arith-to-llvm) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %alloc) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %alloc : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}


// -----// IR Dump After NormalizeMemRefs (normalize-memrefs) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %alloc) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %alloc : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}


// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(128 : index) : i64
    %2 = llvm.mlir.constant(64 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(8192 : index) : i64
    %5 = llvm.mlir.constant(8192 : index) : i64
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.getelementptr %6[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %8 = llvm.ptrtoint %7 : !llvm.ptr to i64
    %9 = llvm.mlir.constant(64 : index) : i64
    %10 = llvm.add %8, %9  : i64
    %11 = llvm.call @malloc(%10) : (i64) -> !llvm.ptr
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.sub %9, %13  : i64
    %15 = llvm.add %12, %14  : i64
    %16 = llvm.urem %15, %9  : i64
    %17 = llvm.sub %15, %16  : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %19 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %20 = llvm.insertvalue %11, %19[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %18, %20[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.mlir.constant(0 : index) : i64
    %23 = llvm.insertvalue %22, %21[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.insertvalue %0, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.insertvalue %1, %24[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.insertvalue %2, %25[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %27 = llvm.insertvalue %4, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %28 = llvm.insertvalue %2, %27[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %29 = llvm.insertvalue %3, %28[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %30 = builtin.unrealized_conversion_cast %29 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %30) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %30 : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}


// -----// IR Dump After ExpandOps (memref-expand) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(128 : index) : i64
    %2 = llvm.mlir.constant(64 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(8192 : index) : i64
    %5 = llvm.mlir.constant(8192 : index) : i64
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.getelementptr %6[8192] : (!llvm.ptr) -> !llvm.ptr, f32
    %8 = llvm.ptrtoint %7 : !llvm.ptr to i64
    %9 = llvm.mlir.constant(64 : index) : i64
    %10 = llvm.add %8, %9  : i64
    %11 = llvm.call @malloc(%10) : (i64) -> !llvm.ptr
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.sub %9, %13  : i64
    %15 = llvm.add %12, %14  : i64
    %16 = llvm.urem %15, %9  : i64
    %17 = llvm.sub %15, %16  : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %19 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %20 = llvm.insertvalue %11, %19[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %18, %20[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.mlir.constant(0 : index) : i64
    %23 = llvm.insertvalue %22, %21[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.insertvalue %0, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.insertvalue %1, %24[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.insertvalue %2, %25[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %27 = llvm.insertvalue %4, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %28 = llvm.insertvalue %2, %27[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %29 = llvm.insertvalue %3, %28[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %30 = builtin.unrealized_conversion_cast %29 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %30) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %30 : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}


// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  func.func @forward(%arg0: memref<1x128x64xf32>) -> memref<1x128x64xf32> {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(128 : index) : i64
    %2 = llvm.mlir.constant(64 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(8192 : index) : i64
    %5 = llvm.mlir.constant(8192 : index) : i64
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.getelementptr %6[8192] : (!llvm.ptr) -> !llvm.ptr, f32
    %8 = llvm.ptrtoint %7 : !llvm.ptr to i64
    %9 = llvm.mlir.constant(64 : index) : i64
    %10 = llvm.add %8, %9  : i64
    %11 = llvm.call @malloc(%10) : (i64) -> !llvm.ptr
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.sub %9, %13  : i64
    %15 = llvm.add %12, %14  : i64
    %16 = llvm.urem %15, %9  : i64
    %17 = llvm.sub %15, %16  : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %19 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %20 = llvm.insertvalue %11, %19[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %18, %20[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.mlir.constant(0 : index) : i64
    %23 = llvm.insertvalue %22, %21[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.insertvalue %0, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.insertvalue %1, %24[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.insertvalue %2, %25[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %27 = llvm.insertvalue %4, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %28 = llvm.insertvalue %2, %27[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %29 = llvm.insertvalue %3, %28[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %30 = builtin.unrealized_conversion_cast %29 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<1x128x64xf32>
    call @forward_kernel_0(%arg0, %30) : (memref<1x128x64xf32>, memref<1x128x64xf32>) -> ()
    return %30 : memref<1x128x64xf32>
  }
  func.func private @forward_kernel_0(memref<1x128x64xf32>, memref<1x128x64xf32>)
}


// -----// IR Dump After ConvertFuncToLLVMPass (convert-func-to-llvm) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @forward(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.mlir.constant(8192 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.mlir.constant(128 : index) : i64
    %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %11 = llvm.mlir.constant(64 : index) : i64
    %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.mlir.constant(64 : index) : i64
    %14 = llvm.insertvalue %13, %12[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.insertvalue %15, %14[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.mlir.constant(128 : index) : i64
    %19 = llvm.mlir.constant(64 : index) : i64
    %20 = llvm.mlir.constant(1 : index) : i64
    %21 = llvm.mlir.constant(8192 : index) : i64
    %22 = llvm.mlir.constant(8192 : index) : i64
    %23 = llvm.mlir.zero : !llvm.ptr
    %24 = llvm.getelementptr %23[8192] : (!llvm.ptr) -> !llvm.ptr, f32
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.mlir.constant(64 : index) : i64
    %27 = llvm.add %25, %26  : i64
    %28 = llvm.call @malloc(%27) : (i64) -> !llvm.ptr
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.sub %26, %30  : i64
    %32 = llvm.add %29, %31  : i64
    %33 = llvm.urem %32, %26  : i64
    %34 = llvm.sub %32, %33  : i64
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr
    %36 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %37 = llvm.insertvalue %28, %36[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %39 = llvm.mlir.constant(0 : index) : i64
    %40 = llvm.insertvalue %39, %38[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %41 = llvm.insertvalue %17, %40[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %42 = llvm.insertvalue %18, %41[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %43 = llvm.insertvalue %19, %42[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %44 = llvm.insertvalue %21, %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %45 = llvm.insertvalue %19, %44[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %46 = llvm.insertvalue %20, %45[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %47 = builtin.unrealized_conversion_cast %46 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<1x128x64xf32>
    %48 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %49 = llvm.extractvalue %46[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.call @forward_kernel_0(%48, %49) : (!llvm.ptr, !llvm.ptr) -> ()
    %50 = llvm.extractvalue %46[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.return %50 : !llvm.ptr
  }
  llvm.func @forward_kernel_0(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}


// -----// IR Dump After ConvertFuncToLLVMPass (convert-func-to-llvm) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @forward(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.mlir.constant(8192 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.mlir.constant(128 : index) : i64
    %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %11 = llvm.mlir.constant(64 : index) : i64
    %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.mlir.constant(64 : index) : i64
    %14 = llvm.insertvalue %13, %12[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.insertvalue %15, %14[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.mlir.constant(128 : index) : i64
    %19 = llvm.mlir.constant(64 : index) : i64
    %20 = llvm.mlir.constant(1 : index) : i64
    %21 = llvm.mlir.constant(8192 : index) : i64
    %22 = llvm.mlir.constant(8192 : index) : i64
    %23 = llvm.mlir.zero : !llvm.ptr
    %24 = llvm.getelementptr %23[8192] : (!llvm.ptr) -> !llvm.ptr, f32
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.mlir.constant(64 : index) : i64
    %27 = llvm.add %25, %26  : i64
    %28 = llvm.call @malloc(%27) : (i64) -> !llvm.ptr
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.sub %26, %30  : i64
    %32 = llvm.add %29, %31  : i64
    %33 = llvm.urem %32, %26  : i64
    %34 = llvm.sub %32, %33  : i64
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr
    %36 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %37 = llvm.insertvalue %28, %36[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %39 = llvm.mlir.constant(0 : index) : i64
    %40 = llvm.insertvalue %39, %38[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %41 = llvm.insertvalue %17, %40[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %42 = llvm.insertvalue %18, %41[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %43 = llvm.insertvalue %19, %42[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %44 = llvm.insertvalue %21, %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %45 = llvm.insertvalue %19, %44[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %46 = llvm.insertvalue %20, %45[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %47 = builtin.unrealized_conversion_cast %46 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<1x128x64xf32>
    %48 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %49 = llvm.extractvalue %46[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.call @forward_kernel_0(%48, %49) : (!llvm.ptr, !llvm.ptr) -> ()
    %50 = llvm.extractvalue %46[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.return %50 : !llvm.ptr
  }
  llvm.func @forward_kernel_0(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}


// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @forward(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.mlir.constant(8192 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.mlir.constant(128 : index) : i64
    %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %11 = llvm.mlir.constant(64 : index) : i64
    %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.mlir.constant(64 : index) : i64
    %14 = llvm.insertvalue %13, %12[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.insertvalue %15, %14[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.mlir.constant(128 : index) : i64
    %19 = llvm.mlir.constant(64 : index) : i64
    %20 = llvm.mlir.constant(1 : index) : i64
    %21 = llvm.mlir.constant(8192 : index) : i64
    %22 = llvm.mlir.constant(8192 : index) : i64
    %23 = llvm.mlir.zero : !llvm.ptr
    %24 = llvm.getelementptr %23[8192] : (!llvm.ptr) -> !llvm.ptr, f32
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.mlir.constant(64 : index) : i64
    %27 = llvm.add %25, %26  : i64
    %28 = llvm.call @malloc(%27) : (i64) -> !llvm.ptr
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.sub %26, %30  : i64
    %32 = llvm.add %29, %31  : i64
    %33 = llvm.urem %32, %26  : i64
    %34 = llvm.sub %32, %33  : i64
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr
    %36 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %37 = llvm.insertvalue %28, %36[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %39 = llvm.mlir.constant(0 : index) : i64
    %40 = llvm.insertvalue %39, %38[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %41 = llvm.insertvalue %17, %40[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %42 = llvm.insertvalue %18, %41[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %43 = llvm.insertvalue %19, %42[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %44 = llvm.insertvalue %21, %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %45 = llvm.insertvalue %19, %44[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %46 = llvm.insertvalue %20, %45[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %47 = builtin.unrealized_conversion_cast %46 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<1x128x64xf32>
    %48 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %49 = llvm.extractvalue %46[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.call @forward_kernel_0(%48, %49) : (!llvm.ptr, !llvm.ptr) -> ()
    %50 = llvm.extractvalue %46[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.return %50 : !llvm.ptr
  }
  llvm.func @forward_kernel_0(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}


// -----// IR Dump After CSE (cse) //----- //
module attributes {torch.debug_module_name = "ReLU"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @forward(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.mlir.constant(8192 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.mlir.constant(128 : index) : i64
    %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %11 = llvm.mlir.constant(64 : index) : i64
    %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.insertvalue %11, %12[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.insertvalue %5, %13[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.mlir.zero : !llvm.ptr
    %16 = llvm.getelementptr %15[8192] : (!llvm.ptr) -> !llvm.ptr, f32
    %17 = llvm.ptrtoint %16 : !llvm.ptr to i64
    %18 = llvm.add %17, %11  : i64
    %19 = llvm.call @malloc(%18) : (i64) -> !llvm.ptr
    %20 = llvm.ptrtoint %19 : !llvm.ptr to i64
    %21 = llvm.sub %11, %5  : i64
    %22 = llvm.add %20, %21  : i64
    %23 = llvm.urem %22, %11  : i64
    %24 = llvm.sub %22, %23  : i64
    %25 = llvm.inttoptr %24 : i64 to !llvm.ptr
    %26 = llvm.insertvalue %19, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %27 = llvm.insertvalue %25, %26[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %28 = llvm.insertvalue %3, %27[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %29 = llvm.insertvalue %5, %28[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %30 = llvm.insertvalue %9, %29[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %31 = llvm.insertvalue %11, %30[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %32 = llvm.insertvalue %7, %31[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %33 = llvm.insertvalue %11, %32[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %34 = llvm.insertvalue %5, %33[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %35 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %36 = llvm.extractvalue %34[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.call @forward_kernel_0(%35, %36) : (!llvm.ptr, !llvm.ptr) -> ()
    %37 = llvm.extractvalue %34[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.return %37 : !llvm.ptr
  }
  llvm.func @forward_kernel_0(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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


// -----// IR Dump After ReconcileUnrealizedCasts (reconcile-unrealized-casts) //----- //
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


