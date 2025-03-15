; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @atax(ptr %0, ptr %1, ptr %2, ptr %3) {
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %9, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 390
  br i1 %7, label %8, label %10

8:                                                ; preds = %5
  call void @atax_kernel_0(ptr %0, i64 %6, ptr %1, ptr %3)
  call void @atax_kernel_1(ptr %3, i64 %6, ptr %2, ptr %0)
  %9 = add i64 %6, 1
  br label %5

10:                                               ; preds = %5
  ret void
}

declare void @atax_kernel_0(ptr, i64, ptr, ptr)

declare void @atax_kernel_1(ptr, i64, ptr, ptr)

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
