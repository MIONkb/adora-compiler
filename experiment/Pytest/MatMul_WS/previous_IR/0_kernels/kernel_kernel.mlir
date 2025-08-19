module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  // func.func @unroll_MATMUL(%arg0: memref<?x36xi32>, %arg1: memref<?x36xi32>, %arg2: memref<?x36xi32>, %arg3: memref<?x36xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
  //   ADORA.kernel {
  //     affine.for %arg4 = 0 to 36 {
  //       affine.for %arg5 = 0 to 36 {
  //         %0 = affine.load %arg2[%arg4, %arg5] : memref<?x36xi32>
  //         affine.store %0, %arg3[%arg4, %arg5] : memref<?x36xi32>
  //         affine.for %arg6 = 0 to 36 {
  //           %1 = affine.load %arg0[%arg4, %arg6] : memref<?x36xi32>
  //           %2 = affine.load %arg1[%arg6, %arg5] : memref<?x36xi32>
  //           %3 = arith.muli %1, %2 : i32
  //           %4 = affine.load %arg3[%arg4, %arg5] : memref<?x36xi32>
  //           %5 = arith.addi %4, %3 : i32
  //           affine.store %5, %arg3[%arg4, %arg5] : memref<?x36xi32>
  //         }
  //       }
  //     }
  //     ADORA.terminator
  //   } {KernelName = "unroll_MATMUL"}
  //   return
  // }
  // func.func @merge_MATMUL(%arg0: memref<?x36xi32>, %arg1: memref<?x36xi32>, %arg2: memref<?x36xi32>, %arg3: memref<?x36xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    // ADORA.kernel {
      // affine.for %arg4 = 0 to 36 {
        // %0 = affine.load %arg2[%arg4, 0] : memref<?x36xi32>
        // affine.store %0, %arg3[%arg4, 0] : memref<?x36xi32>
        // %1 = affine.load %arg2[%arg4, 1] : memref<?x36xi32>
        // affine.store %1, %arg3[%arg4, 1] : memref<?x36xi32>
        // %2 = affine.load %arg2[%arg4, 2] : memref<?x36xi32>
        // affine.store %2, %arg3[%arg4, 2] : memref<?x36xi32>
        // %3 = affine.load %arg2[%arg4, 3] : memref<?x36xi32>
        // affine.store %3, %arg3[%arg4, 3] : memref<?x36xi32>
        // affine.for %arg5 = 0 to 36 {
          // %4 = affine.load %arg0[%arg4, %arg5] : memref<?x36xi32>
          // %5 = affine.load %arg1[%arg5, 0] : memref<?x36xi32>
          // %6 = arith.muli %4, %5 : i32
          // %7 = affine.load %arg3[%arg4, 0] : memref<?x36xi32>
          // %8 = arith.addi %7, %6 : i32
          // affine.store %8, %arg3[%arg4, 0] : memref<?x36xi32>
          // %9 = affine.load %arg0[%arg4, %arg5] : memref<?x36xi32>
          // %10 = affine.load %arg1[%arg5, 1] : memref<?x36xi32>
          // %11 = arith.muli %9, %10 : i32
          // %12 = affine.load %arg3[%arg4, 1] : memref<?x36xi32>
          // %13 = arith.addi %12, %11 : i32
          // affine.store %13, %arg3[%arg4, 1] : memref<?x36xi32>
          // %14 = affine.load %arg0[%arg4, %arg5] : memref<?x36xi32>
          // %15 = affine.load %arg1[%arg5, 2] : memref<?x36xi32>
          // %16 = arith.muli %14, %15 : i32
          // %17 = affine.load %arg3[%arg4, 2] : memref<?x36xi32>
          // %18 = arith.addi %17, %16 : i32
          // affine.store %18, %arg3[%arg4, 2] : memref<?x36xi32>
          // %19 = affine.load %arg0[%arg4, %arg5] : memref<?x36xi32>
          // %20 = affine.load %arg1[%arg5, 3] : memref<?x36xi32>
          // %21 = arith.muli %19, %20 : i32
          // %22 = affine.load %arg3[%arg4, 3] : memref<?x36xi32>
          // %23 = arith.addi %22, %21 : i32
          // affine.store %23, %arg3[%arg4, 3] : memref<?x36xi32>
        // }
      // }
      // ADORA.terminator
    // } {KernelName = "merge_MATMUL"}
    // return
  // }
  func.func @merge_MATMUL_4x4(%arg0: memref<?x36xi32>, %arg1: memref<?x36xi32>, %arg2: memref<?x36xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    ADORA.kernel {
      affine.for %arg3 = 0 to 9 {
        affine.for %arg4 = 0 to 9 {
          affine.store %c0_i32, %arg2[%arg3 * 4, %arg4 * 4] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4, %arg4 * 4 + 1] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4, %arg4 * 4 + 2] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4, %arg4 * 4 + 3] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4 + 1, %arg4 * 4] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4 + 1, %arg4 * 4 + 1] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4 + 1, %arg4 * 4 + 2] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4 + 1, %arg4 * 4 + 3] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4 + 2, %arg4 * 4] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4 + 2, %arg4 * 4 + 1] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4 + 2, %arg4 * 4 + 2] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4 + 2, %arg4 * 4 + 3] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4 + 3, %arg4 * 4] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4 + 3, %arg4 * 4 + 1] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4 + 3, %arg4 * 4 + 2] : memref<?x36xi32>
          affine.store %c0_i32, %arg2[%arg3 * 4 + 3, %arg4 * 4 + 3] : memref<?x36xi32>
          affine.for %arg5 = 0 to 36 {
            %0 = affine.load %arg0[%arg3 * 4, %arg5] : memref<?x36xi32>
            %1 = affine.load %arg1[%arg5, %arg4 * 4] : memref<?x36xi32>
            %2 = arith.muli %0, %1 : i32
            %3 = affine.load %arg2[%arg3 * 4, %arg4 * 4] : memref<?x36xi32>
            %4 = arith.addi %3, %2 : i32
            affine.store %4, %arg2[%arg3 * 4, %arg4 * 4] : memref<?x36xi32>
            %5 = affine.load %arg0[%arg3 * 4, %arg5] : memref<?x36xi32>
            %6 = affine.load %arg1[%arg5, %arg4 * 4 + 1] : memref<?x36xi32>
            %7 = arith.muli %5, %6 : i32
            %8 = affine.load %arg2[%arg3 * 4, %arg4 * 4 + 1] : memref<?x36xi32>
            %9 = arith.addi %8, %7 : i32
            affine.store %9, %arg2[%arg3 * 4, %arg4 * 4 + 1] : memref<?x36xi32>
            %10 = affine.load %arg0[%arg3 * 4, %arg5] : memref<?x36xi32>
            %11 = affine.load %arg1[%arg5, %arg4 * 4 + 2] : memref<?x36xi32>
            %12 = arith.muli %10, %11 : i32
            %13 = affine.load %arg2[%arg3 * 4, %arg4 * 4 + 2] : memref<?x36xi32>
            %14 = arith.addi %13, %12 : i32
            affine.store %14, %arg2[%arg3 * 4, %arg4 * 4 + 2] : memref<?x36xi32>
            %15 = affine.load %arg0[%arg3 * 4, %arg5] : memref<?x36xi32>
            %16 = affine.load %arg1[%arg5, %arg4 * 4 + 3] : memref<?x36xi32>
            %17 = arith.muli %15, %16 : i32
            %18 = affine.load %arg2[%arg3 * 4, %arg4 * 4 + 3] : memref<?x36xi32>
            %19 = arith.addi %18, %17 : i32
            affine.store %19, %arg2[%arg3 * 4, %arg4 * 4 + 3] : memref<?x36xi32>
            %20 = affine.load %arg0[%arg3 * 4 + 1, %arg5] : memref<?x36xi32>
            %21 = affine.load %arg1[%arg5, %arg4 * 4] : memref<?x36xi32>
            %22 = arith.muli %20, %21 : i32
            %23 = affine.load %arg2[%arg3 * 4 + 1, %arg4 * 4] : memref<?x36xi32>
            %24 = arith.addi %23, %22 : i32
            affine.store %24, %arg2[%arg3 * 4 + 1, %arg4 * 4] : memref<?x36xi32>
            %25 = affine.load %arg0[%arg3 * 4 + 1, %arg5] : memref<?x36xi32>
            %26 = affine.load %arg1[%arg5, %arg4 * 4 + 1] : memref<?x36xi32>
            %27 = arith.muli %25, %26 : i32
            %28 = affine.load %arg2[%arg3 * 4 + 1, %arg4 * 4 + 1] : memref<?x36xi32>
            %29 = arith.addi %28, %27 : i32
            affine.store %29, %arg2[%arg3 * 4 + 1, %arg4 * 4 + 1] : memref<?x36xi32>
            %30 = affine.load %arg0[%arg3 * 4 + 1, %arg5] : memref<?x36xi32>
            %31 = affine.load %arg1[%arg5, %arg4 * 4 + 2] : memref<?x36xi32>
            %32 = arith.muli %30, %31 : i32
            %33 = affine.load %arg2[%arg3 * 4 + 1, %arg4 * 4 + 2] : memref<?x36xi32>
            %34 = arith.addi %33, %32 : i32
            affine.store %34, %arg2[%arg3 * 4 + 1, %arg4 * 4 + 2] : memref<?x36xi32>
            %35 = affine.load %arg0[%arg3 * 4 + 1, %arg5] : memref<?x36xi32>
            %36 = affine.load %arg1[%arg5, %arg4 * 4 + 3] : memref<?x36xi32>
            %37 = arith.muli %35, %36 : i32
            %38 = affine.load %arg2[%arg3 * 4 + 1, %arg4 * 4 + 3] : memref<?x36xi32>
            %39 = arith.addi %38, %37 : i32
            affine.store %39, %arg2[%arg3 * 4 + 1, %arg4 * 4 + 3] : memref<?x36xi32>
            %40 = affine.load %arg0[%arg3 * 4 + 2, %arg5] : memref<?x36xi32>
            %41 = affine.load %arg1[%arg5, %arg4 * 4] : memref<?x36xi32>
            %42 = arith.muli %40, %41 : i32
            %43 = affine.load %arg2[%arg3 * 4 + 2, %arg4 * 4] : memref<?x36xi32>
            %44 = arith.addi %43, %42 : i32
            affine.store %44, %arg2[%arg3 * 4 + 2, %arg4 * 4] : memref<?x36xi32>
            %45 = affine.load %arg0[%arg3 * 4 + 2, %arg5] : memref<?x36xi32>
            %46 = affine.load %arg1[%arg5, %arg4 * 4 + 1] : memref<?x36xi32>
            %47 = arith.muli %45, %46 : i32
            %48 = affine.load %arg2[%arg3 * 4 + 2, %arg4 * 4 + 1] : memref<?x36xi32>
            %49 = arith.addi %48, %47 : i32
            affine.store %49, %arg2[%arg3 * 4 + 2, %arg4 * 4 + 1] : memref<?x36xi32>
            %50 = affine.load %arg0[%arg3 * 4 + 2, %arg5] : memref<?x36xi32>
            %51 = affine.load %arg1[%arg5, %arg4 * 4 + 2] : memref<?x36xi32>
            %52 = arith.muli %50, %51 : i32
            %53 = affine.load %arg2[%arg3 * 4 + 2, %arg4 * 4 + 2] : memref<?x36xi32>
            %54 = arith.addi %53, %52 : i32
            affine.store %54, %arg2[%arg3 * 4 + 2, %arg4 * 4 + 2] : memref<?x36xi32>
            %55 = affine.load %arg0[%arg3 * 4 + 2, %arg5] : memref<?x36xi32>
            %56 = affine.load %arg1[%arg5, %arg4 * 4 + 3] : memref<?x36xi32>
            %57 = arith.muli %55, %56 : i32
            %58 = affine.load %arg2[%arg3 * 4 + 2, %arg4 * 4 + 3] : memref<?x36xi32>
            %59 = arith.addi %58, %57 : i32
            affine.store %59, %arg2[%arg3 * 4 + 2, %arg4 * 4 + 3] : memref<?x36xi32>
            %60 = affine.load %arg0[%arg3 * 4 + 3, %arg5] : memref<?x36xi32>
            %61 = affine.load %arg1[%arg5, %arg4 * 4] : memref<?x36xi32>
            %62 = arith.muli %60, %61 : i32
            %63 = affine.load %arg2[%arg3 * 4 + 3, %arg4 * 4] : memref<?x36xi32>
            %64 = arith.addi %63, %62 : i32
            affine.store %64, %arg2[%arg3 * 4 + 3, %arg4 * 4] : memref<?x36xi32>
            %65 = affine.load %arg0[%arg3 * 4 + 3, %arg5] : memref<?x36xi32>
            %66 = affine.load %arg1[%arg5, %arg4 * 4 + 1] : memref<?x36xi32>
            %67 = arith.muli %65, %66 : i32
            %68 = affine.load %arg2[%arg3 * 4 + 3, %arg4 * 4 + 1] : memref<?x36xi32>
            %69 = arith.addi %68, %67 : i32
            affine.store %69, %arg2[%arg3 * 4 + 3, %arg4 * 4 + 1] : memref<?x36xi32>
            %70 = affine.load %arg0[%arg3 * 4 + 3, %arg5] : memref<?x36xi32>
            %71 = affine.load %arg1[%arg5, %arg4 * 4 + 2] : memref<?x36xi32>
            %72 = arith.muli %70, %71 : i32
            %73 = affine.load %arg2[%arg3 * 4 + 3, %arg4 * 4 + 2] : memref<?x36xi32>
            %74 = arith.addi %73, %72 : i32
            affine.store %74, %arg2[%arg3 * 4 + 3, %arg4 * 4 + 2] : memref<?x36xi32>
            %75 = affine.load %arg0[%arg3 * 4 + 3, %arg5] : memref<?x36xi32>
            %76 = affine.load %arg1[%arg5, %arg4 * 4 + 3] : memref<?x36xi32>
            %77 = arith.muli %75, %76 : i32
            %78 = affine.load %arg2[%arg3 * 4 + 3, %arg4 * 4 + 3] : memref<?x36xi32>
            %79 = arith.addi %78, %77 : i32
            affine.store %79, %arg2[%arg3 * 4 + 3, %arg4 * 4 + 3] : memref<?x36xi32>
          }
        }
      }
      ADORA.terminator
    } {KernelName = "merge_MATMUL_4x4"}
    return
  }
  // func.func @merge_MATMUL_4x4_affine(%arg0: memref<?x36xi32>, %arg1: memref<?x36xi32>, %arg2: memref<?x36xi32>, %arg3: memref<?x36xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    // %c36 = arith.constant 36 : index
    // %c0 = arith.constant 0 : index
    // %c4 = arith.constant 4 : index
    // %c0_i32 = arith.constant 0 : i32
    // ADORA.kernel {
      // affine.for %arg4 = 0 to 36 {
        // scf.for %arg5 = %c0 to %c36 step %c4 {
          // affine.store %c0_i32, %arg2[%arg4, 0] : memref<?x36xi32>
          // affine.store %c0_i32, %arg2[%arg4, 1] : memref<?x36xi32>
          // affine.store %c0_i32, %arg2[%arg4, 2] : memref<?x36xi32>
          // affine.store %c0_i32, %arg2[%arg4, 3] : memref<?x36xi32>
          // affine.for %arg6 = 0 to 36 {
            // %0 = affine.load %arg0[%arg4, %arg6] : memref<?x36xi32>
            // %1 = affine.load %arg1[%arg6, 0] : memref<?x36xi32>
            // %2 = arith.muli %0, %1 : i32
            // %3 = affine.load %arg2[%arg4, 0] : memref<?x36xi32>
            // %4 = arith.addi %3, %2 : i32
            // affine.store %4, %arg2[%arg4, 0] : memref<?x36xi32>
            // %5 = affine.load %arg0[%arg4, %arg6] : memref<?x36xi32>
            // %6 = affine.load %arg1[%arg6, 1] : memref<?x36xi32>
            // %7 = arith.muli %5, %6 : i32
            // %8 = affine.load %arg2[%arg4, 1] : memref<?x36xi32>
            // %9 = arith.addi %8, %7 : i32
            // affine.store %9, %arg2[%arg4, 1] : memref<?x36xi32>
            // %10 = affine.load %arg0[%arg4, %arg6] : memref<?x36xi32>
            // %11 = affine.load %arg1[%arg6, 2] : memref<?x36xi32>
            // %12 = arith.muli %10, %11 : i32
            // %13 = affine.load %arg2[%arg4, 2] : memref<?x36xi32>
            // %14 = arith.addi %13, %12 : i32
            // affine.store %14, %arg2[%arg4, 2] : memref<?x36xi32>
            // %15 = affine.load %arg0[%arg4, %arg6] : memref<?x36xi32>
            // %16 = affine.load %arg1[%arg6, 3] : memref<?x36xi32>
            // %17 = arith.muli %15, %16 : i32
            // %18 = affine.load %arg2[%arg4, 3] : memref<?x36xi32>
            // %19 = arith.addi %18, %17 : i32
            // affine.store %19, %arg2[%arg4, 3] : memref<?x36xi32>
          // }
        // }
      // }
      // ADORA.terminator
    // } {KernelName = "merge_MATMUL_4x4_affine"}
    // return
  // }
}

