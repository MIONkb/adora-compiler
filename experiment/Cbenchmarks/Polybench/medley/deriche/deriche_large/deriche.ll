; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @deriche(ptr %0, ptr %1, ptr %2, ptr %3) {
  %5 = alloca float, i64 1, align 4
  store float undef, ptr %5, align 4
  %6 = alloca float, i64 1, align 4
  store float undef, ptr %6, align 4
  %7 = alloca float, i64 1, align 4
  store float undef, ptr %7, align 4
  %8 = alloca float, i64 1, align 4
  store float undef, ptr %8, align 4
  %9 = alloca float, i64 1, align 4
  store float undef, ptr %9, align 4
  %10 = alloca float, i64 1, align 4
  store float undef, ptr %10, align 4
  %11 = alloca float, i64 1, align 4
  store float undef, ptr %11, align 4
  %12 = alloca float, i64 1, align 4
  store float undef, ptr %12, align 4
  %13 = alloca float, i64 1, align 4
  store float undef, ptr %13, align 4
  %14 = alloca float, i64 1, align 4
  store float undef, ptr %14, align 4
  call void @deriche_kernel_0(ptr %12, ptr %11, ptr %14, ptr %0, ptr %2)
  call void @deriche_kernel_1(ptr %6, ptr %5, ptr %10, ptr %9, ptr %3, ptr %0)
  call void @deriche_kernel_2(ptr %2, ptr %3, ptr %1)
  call void @deriche_kernel_3(ptr %13, ptr %12, ptr %11, ptr %1, ptr %2)
  call void @deriche_kernel_4(ptr %8, ptr %7, ptr %6, ptr %5, ptr %3, ptr %1)
  call void @deriche_kernel_5(ptr %2, ptr %3, ptr %1)
  ret void
}

declare void @deriche_kernel_0(ptr, ptr, ptr, ptr, ptr)

declare void @deriche_kernel_1(ptr, ptr, ptr, ptr, ptr, ptr)

declare void @deriche_kernel_2(ptr, ptr, ptr)

declare void @deriche_kernel_3(ptr, ptr, ptr, ptr, ptr)

declare void @deriche_kernel_4(ptr, ptr, ptr, ptr, ptr, ptr)

declare void @deriche_kernel_5(ptr, ptr, ptr)

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
