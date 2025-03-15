; ModuleID = '/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/linear-algebra/blas/gesummv_medium/forward.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "riscv64"

define void @gesummv(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4) local_unnamed_addr #0 {
  tail call void @gesummv_kernel_0(ptr %2, ptr %4, ptr %0, ptr %3, ptr %1)
  ret void
}

declare void @gesummv_kernel_0(ptr, ptr, ptr, ptr, ptr) local_unnamed_addr #0

attributes #0 = { "target-cpu"="rocket-rv64" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
