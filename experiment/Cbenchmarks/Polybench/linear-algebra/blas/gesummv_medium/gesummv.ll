; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @gesummv(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4) {
  call void @gesummv_kernel_0(ptr %2, ptr %4, ptr %0, ptr %3, ptr %1)
  ret void
}

declare void @gesummv_kernel_0(ptr, ptr, ptr, ptr, ptr)

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
