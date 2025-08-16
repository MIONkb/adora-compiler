module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @IntVecAdd(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = ADORA.BlockLoad %arg0 [0] : memref<?xi32> -> memref<20xi32>, stride [2, 2] : {Id = "0", KernelName = "IntVecAdd"}
    %1 = ADORA.BlockLoad %arg1 [0] : memref<?xi32> -> memref<20xi32>  {Id = "1", KernelName = "IntVecAdd"}
    %2 = ADORA.LocalMemAlloc memref<20xi32>  {Id = "2", KernelName = "IntVecAdd"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 20 {
        %3 = affine.load %0[%arg3] : memref<20xi32>
        %4 = affine.load %1[%arg3] : memref<20xi32>
        %5 = arith.addi %3, %4 : i32
        affine.store %5, %2[%arg3] : memref<20xi32>
      }
      ADORA.terminator
    } {KernelName = "IntVecAdd"}
    ADORA.BlockStore %2, %arg2 [0] : memref<20xi32> -> memref<?xi32>  {Id = "2", KernelName = "IntVecAdd"}
    return
  }
}

