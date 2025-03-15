; ModuleID = '/home/jhlou/CGRVOPT/cgra-opt/experiment/AILayers/relu/forward.ll'
source_filename = "LLVMDialectModule"
target triple = "riscv64"

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #0

define ptr @forward(ptr %0) local_unnamed_addr #1 {
  %2 = tail call dereferenceable_or_null(32832) ptr @malloc(i64 32832)
  %3 = ptrtoint ptr %2 to i64
  %4 = add i64 %3, 63
  %5 = and i64 %4, -64
  %6 = inttoptr i64 %5 to ptr
  tail call void @forward_kernel_0(ptr %0, ptr %6)
  ret ptr %2
}

declare void @forward_kernel_0(ptr, ptr) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "target-cpu"="rocket-rv64" }
attributes #1 = { "target-cpu"="rocket-rv64" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
