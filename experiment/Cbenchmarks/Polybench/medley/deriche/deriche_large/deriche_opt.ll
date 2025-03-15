; ModuleID = '/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/medley/deriche/deriche.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "riscv64"

define void @deriche(ptr %0, ptr %1, ptr %2, ptr %3) local_unnamed_addr #0 {
  %5 = alloca float, align 4
  %6 = alloca float, align 4
  %7 = alloca float, align 4
  %8 = alloca float, align 4
  %9 = alloca float, align 4
  %10 = alloca float, align 4
  %11 = alloca float, align 4
  %12 = alloca float, align 4
  %13 = alloca float, align 4
  %14 = alloca float, align 4
  call void @deriche_kernel_0(ptr nonnull %12, ptr nonnull %11, ptr nonnull %14, ptr %0, ptr %2)
  call void @deriche_kernel_1(ptr nonnull %6, ptr nonnull %5, ptr nonnull %10, ptr nonnull %9, ptr %3, ptr %0)
  call void @deriche_kernel_2(ptr %2, ptr %3, ptr %1)
  call void @deriche_kernel_3(ptr nonnull %13, ptr nonnull %12, ptr nonnull %11, ptr %1, ptr %2)
  call void @deriche_kernel_4(ptr nonnull %8, ptr nonnull %7, ptr nonnull %6, ptr nonnull %5, ptr %3, ptr %1)
  call void @deriche_kernel_5(ptr %2, ptr %3, ptr %1)
  ret void
}

declare void @deriche_kernel_0(ptr, ptr, ptr, ptr, ptr) local_unnamed_addr #0

declare void @deriche_kernel_1(ptr, ptr, ptr, ptr, ptr, ptr) local_unnamed_addr #0

declare void @deriche_kernel_2(ptr, ptr, ptr) local_unnamed_addr #0

declare void @deriche_kernel_3(ptr, ptr, ptr, ptr, ptr) local_unnamed_addr #0

declare void @deriche_kernel_4(ptr, ptr, ptr, ptr, ptr, ptr) local_unnamed_addr #0

declare void @deriche_kernel_5(ptr, ptr, ptr) local_unnamed_addr #0

attributes #0 = { "target-cpu"="rocket-rv64" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
