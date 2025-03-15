; ModuleID = '/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/stencils/jacobi-1d-large/jacobi-1d.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "riscv64"

define void @jacobi_1d(ptr %0, ptr %1) local_unnamed_addr #0 {
  br label %3

3:                                                ; preds = %2, %3
  %4 = phi i64 [ 0, %2 ], [ %5, %3 ]
  tail call void @jacobi_1d_kernel_0(ptr %0, ptr %1)
  tail call void @jacobi_1d_kernel_1(ptr %1, ptr %0)
  %5 = add nuw nsw i64 %4, 1
  %exitcond.not = icmp eq i64 %5, 500
  br i1 %exitcond.not, label %6, label %3

6:                                                ; preds = %3
  ret void
}

declare void @jacobi_1d_kernel_0(ptr, ptr) local_unnamed_addr #0

declare void @jacobi_1d_kernel_1(ptr, ptr) local_unnamed_addr #0

attributes #0 = { "target-cpu"="rocket-rv64" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
