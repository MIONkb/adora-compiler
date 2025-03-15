; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree nounwind willreturn memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64) #0

declare void @free(ptr) #1

define void @bicg(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4) #1 {
  br label %6

6:                                                ; preds = %9, %5
  %7 = phi i64 [ 0, %5 ], [ %11, %9 ]
  %8 = icmp slt i64 %7, 390
  br i1 %8, label %9, label %12

9:                                                ; preds = %6
  %10 = getelementptr float, ptr %1, i64 %7
  store float 0.000000e+00, ptr %10, align 4
  %11 = add i64 %7, 1
  br label %6

12:                                               ; preds = %6
  br label %13

13:                                               ; preds = %41, %12
  %14 = phi i64 [ 0, %12 ], [ %42, %41 ]
  %15 = icmp slt i64 %14, 410
  br i1 %15, label %16, label %43

16:                                               ; preds = %13
  %17 = getelementptr float, ptr %2, i64 %14
  store float 0.000000e+00, ptr %17, align 4
  br label %18

18:                                               ; preds = %21, %16
  %19 = phi i64 [ 0, %16 ], [ %40, %21 ]
  %20 = icmp slt i64 %19, 390
  br i1 %20, label %21, label %41

21:                                               ; preds = %18
  %22 = getelementptr float, ptr %1, i64 %19
  %23 = load float, ptr %22, align 4
  %24 = getelementptr float, ptr %4, i64 %14
  %25 = load float, ptr %24, align 4
  %26 = getelementptr [390 x float], ptr %0, i64 %14, i64 %19
  %27 = load float, ptr %26, align 4
  %28 = fmul float %25, %27
  %29 = fadd float %23, %28
  %30 = getelementptr float, ptr %1, i64 %19
  store float %29, ptr %30, align 4
  %31 = getelementptr float, ptr %2, i64 %14
  %32 = load float, ptr %31, align 4
  %33 = getelementptr [390 x float], ptr %0, i64 %14, i64 %19
  %34 = load float, ptr %33, align 4
  %35 = getelementptr float, ptr %3, i64 %19
  %36 = load float, ptr %35, align 4
  %37 = fmul float %34, %36
  %38 = fadd float %32, %37
  %39 = getelementptr float, ptr %2, i64 %14
  store float %38, ptr %39, align 4
  %40 = add i64 %19, 1
  br label %18

41:                                               ; preds = %18
  %42 = add i64 %14, 1
  br label %13

43:                                               ; preds = %13
  ret void
}

attributes #0 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: readwrite) "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

