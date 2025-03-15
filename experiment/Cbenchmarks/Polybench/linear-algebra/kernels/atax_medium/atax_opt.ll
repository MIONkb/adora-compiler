; ModuleID = '/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/linear-algebra/kernels/atax_medium/atax.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "riscv64"

define void @atax(ptr %0, ptr %1, ptr %2, ptr %3) local_unnamed_addr #0 {
  br label %5

5:                                                ; preds = %4, %5
  %6 = phi i64 [ 0, %4 ], [ %7, %5 ]
  tail call void @atax_kernel_0(ptr %0, i64 %6, ptr %1, ptr %3)
  tail call void @atax_kernel_1(ptr %3, i64 %6, ptr %2, ptr %0)
  %7 = add nuw nsw i64 %6, 1
  %exitcond.not = icmp eq i64 %7, 390
  br i1 %exitcond.not, label %8, label %5

8:                                                ; preds = %5
  ret void
}

declare void @atax_kernel_0(ptr, i64, ptr, ptr) local_unnamed_addr #0

declare void @atax_kernel_1(ptr, i64, ptr, ptr) local_unnamed_addr #0

attributes #0 = { "target-cpu"="rocket-rv64" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
