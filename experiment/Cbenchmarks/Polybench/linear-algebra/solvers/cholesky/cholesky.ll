; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @cholesky_kernel_0(ptr, i64, i64)

declare void @cholesky_kernel_1(ptr, i64)

define void @cholesky(ptr %0) {
  br label %2

2:                                                ; preds = %20, %1
  %3 = phi i64 [ %26, %20 ], [ 0, %1 ]
  %4 = icmp slt i64 %3, 2000
  br i1 %4, label %5, label %27

5:                                                ; preds = %2
  br label %6

6:                                                ; preds = %9, %5
  %7 = phi i64 [ %19, %9 ], [ 0, %5 ]
  %8 = icmp slt i64 %7, %3
  br i1 %8, label %9, label %20

9:                                                ; preds = %6
  call void @cholesky_kernel_0(ptr %0, i64 %3, i64 %7)
  %10 = mul i64 %7, 2000
  %11 = add i64 %10, %7
  %12 = getelementptr float, ptr %0, i64 %11
  %13 = load float, ptr %12, align 4
  %14 = mul i64 %3, 2000
  %15 = add i64 %14, %7
  %16 = getelementptr float, ptr %0, i64 %15
  %17 = load float, ptr %16, align 4
  %18 = fdiv float %17, %13
  store float %18, ptr %16, align 4
  %19 = add i64 %7, 1
  br label %6

20:                                               ; preds = %6
  call void @cholesky_kernel_1(ptr %0, i64 %3)
  %21 = mul i64 %3, 2000
  %22 = add i64 %21, %3
  %23 = getelementptr float, ptr %0, i64 %22
  %24 = load float, ptr %23, align 4
  %25 = call float @llvm.sqrt.f32(float %24)
  store float %25, ptr %23, align 4
  %26 = add i64 %3, 1
  br label %2

27:                                               ; preds = %2
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
