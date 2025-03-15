; ModuleID = '/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/linear-algebra/solvers/cholesky/cholesky.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "riscv64"

declare void @cholesky_kernel_0(ptr, i64, i64) local_unnamed_addr #0

declare void @cholesky_kernel_1(ptr, i64) local_unnamed_addr #0

define void @cholesky(ptr %0) local_unnamed_addr #0 {
  br label %.preheader

.preheader:                                       ; preds = %1, %._crit_edge
  %2 = phi i64 [ 0, %1 ], [ %18, %._crit_edge ]
  %.not = icmp eq i64 %2, 0
  br i1 %.not, label %._crit_edge, label %.lr.ph

.lr.ph:                                           ; preds = %.preheader
  %3 = mul nuw nsw i64 %2, 2000
  br label %4

4:                                                ; preds = %.lr.ph, %4
  %5 = phi i64 [ 0, %.lr.ph ], [ %13, %4 ]
  tail call void @cholesky_kernel_0(ptr %0, i64 %2, i64 %5)
  %6 = mul nuw nsw i64 %5, 2001
  %7 = getelementptr float, ptr %0, i64 %6
  %8 = load float, ptr %7, align 4
  %9 = add nuw nsw i64 %5, %3
  %10 = getelementptr float, ptr %0, i64 %9
  %11 = load float, ptr %10, align 4
  %12 = fdiv float %11, %8
  store float %12, ptr %10, align 4
  %13 = add nuw nsw i64 %5, 1
  %exitcond.not = icmp eq i64 %13, %2
  br i1 %exitcond.not, label %._crit_edge, label %4

._crit_edge:                                      ; preds = %4, %.preheader
  tail call void @cholesky_kernel_1(ptr %0, i64 %2)
  %14 = mul nuw nsw i64 %2, 2001
  %15 = getelementptr float, ptr %0, i64 %14
  %16 = load float, ptr %15, align 4
  %17 = tail call float @llvm.sqrt.f32(float %16)
  store float %17, ptr %15, align 4
  %18 = add nuw nsw i64 %2, 1
  %exitcond2.not = icmp eq i64 %18, 2000
  br i1 %exitcond2.not, label %19, label %.preheader

19:                                               ; preds = %._crit_edge
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

attributes #0 = { "target-cpu"="rocket-rv64" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) "target-cpu"="rocket-rv64" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
