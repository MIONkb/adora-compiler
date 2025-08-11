module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @ss_sort(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.for %arg4 = %c0 to %c32 step %c2 iter_args(%arg5 = %c0_i32) -> (i32) {
      %1 = arith.index_cast %arg4 : index to i32
      func.call @init(%arg2) : (memref<?xi32>) -> ()
      %2 = arith.cmpi eq, %arg5, %c0_i32 : i32
      scf.if %2 {
        func.call @hist(%arg2, %arg0, %1) : (memref<?xi32>, memref<?xi32>, i32) -> ()
      } else {
        func.call @hist(%arg2, %arg1, %1) : (memref<?xi32>, memref<?xi32>, i32) -> ()
      }
      func.call @local_scan(%arg2) : (memref<?xi32>) -> ()
      func.call @sum_scan(%arg3, %arg2) : (memref<?xi32>, memref<?xi32>) -> ()
      func.call @last_step_scan(%arg2, %arg3) : (memref<?xi32>, memref<?xi32>) -> ()
      %3 = arith.extui %2 : i1 to i32
      scf.if %2 {
        func.call @update(%arg1, %arg2, %arg0, %1) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> ()
      } else {
        func.call @update(%arg0, %arg2, %arg1, %1) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> ()
      }
      scf.yield %3 : i32
    }
    return
  }
  func.func @init(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<available_externally>} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg1 = 0 to 2048 {
      affine.store %c0_i32, %arg0[%arg1] : memref<?xi32>
    }
    return
  }
  func.func @hist(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<available_externally>} {
    %c1_i32 = arith.constant 1 : i32
    %c512_i32 = arith.constant 512 : i32
    %c3_i32 = arith.constant 3 : i32
    affine.for %arg3 = 0 to 512 {
      %0 = arith.index_cast %arg3 : index to i32
      affine.for %arg4 = 0 to 4 {
        %1 = affine.load %arg1[%arg4 + %arg3 * 4] : memref<?xi32>
        %2 = arith.shrsi %1, %arg2 : i32
        %3 = arith.andi %2, %c3_i32 : i32
        %4 = arith.muli %3, %c512_i32 : i32
        %5 = arith.addi %4, %0 : i32
        %6 = arith.addi %5, %c1_i32 : i32
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %arg0[%7] : memref<?xi32>
        %9 = arith.addi %8, %c1_i32 : i32
        memref.store %9, %arg0[%7] : memref<?xi32>
      }
    }
    return
  }
  func.func @local_scan(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<available_externally>} {
    affine.for %arg1 = 0 to 128 {
      affine.for %arg2 = 1 to 16 {
        %0 = affine.load %arg0[%arg2 + %arg1 * 16 - 1] : memref<?xi32>
        %1 = affine.load %arg0[%arg2 + %arg1 * 16] : memref<?xi32>
        %2 = arith.addi %1, %0 : i32
        affine.store %2, %arg0[%arg2 + %arg1 * 16] : memref<?xi32>
      }
    }
    return
  }
  func.func @sum_scan(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<available_externally>} {
    %c0_i32 = arith.constant 0 : i32
    affine.store %c0_i32, %arg0[0] : memref<?xi32>
    affine.for %arg2 = 1 to 128 {
      %0 = affine.load %arg0[%arg2 - 1] : memref<?xi32>
      %1 = affine.load %arg1[%arg2 * 16 - 1] : memref<?xi32>
      %2 = arith.addi %0, %1 : i32
      affine.store %2, %arg0[%arg2] : memref<?xi32>
    }
    return
  }
  func.func @last_step_scan(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<available_externally>} {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 16 {
        %0 = affine.load %arg0[%arg3 + %arg2 * 16] : memref<?xi32>
        %1 = affine.load %arg1[%arg2] : memref<?xi32>
        %2 = arith.addi %0, %1 : i32
        affine.store %2, %arg0[%arg3 + %arg2 * 16] : memref<?xi32>
      }
    }
    return
  }
  func.func @update(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<available_externally>} {
    %c1_i32 = arith.constant 1 : i32
    %c512_i32 = arith.constant 512 : i32
    %c3_i32 = arith.constant 3 : i32
    affine.for %arg4 = 0 to 512 {
      %0 = arith.index_cast %arg4 : index to i32
      affine.for %arg5 = 0 to 4 {
        %1 = affine.load %arg2[%arg5 + %arg4 * 4] : memref<?xi32>
        %2 = arith.shrsi %1, %arg3 : i32
        %3 = arith.andi %2, %c3_i32 : i32
        %4 = arith.muli %3, %c512_i32 : i32
        %5 = arith.addi %4, %0 : i32
        %6 = arith.index_cast %5 : i32 to index
        %7 = memref.load %arg1[%6] : memref<?xi32>
        %8 = arith.index_cast %7 : i32 to index
        memref.store %1, %arg0[%8] : memref<?xi32>
        %9 = memref.load %arg1[%6] : memref<?xi32>
        %10 = arith.addi %9, %c1_i32 : i32
        memref.store %10, %arg1[%6] : memref<?xi32>
      }
    }
    return
  }
}
