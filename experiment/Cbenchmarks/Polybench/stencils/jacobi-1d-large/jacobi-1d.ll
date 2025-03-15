; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @jacobi_1d(ptr %0, ptr %1) {
  br label %3

3:                                                ; preds = %6, %2
  %4 = phi i64 [ %7, %6 ], [ 0, %2 ]
  %5 = icmp slt i64 %4, 500
  br i1 %5, label %6, label %8

6:                                                ; preds = %3
  call void @jacobi_1d_kernel_0(ptr %0, ptr %1)
  call void @jacobi_1d_kernel_1(ptr %1, ptr %0)
  %7 = add i64 %4, 1
  br label %3

8:                                                ; preds = %3
  ret void
}

declare void @jacobi_1d_kernel_0(ptr, ptr)

declare void @jacobi_1d_kernel_1(ptr, ptr)

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
