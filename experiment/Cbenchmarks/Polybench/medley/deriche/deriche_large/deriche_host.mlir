module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @deriche(%arg0: memref<4096x2160xf32>, %arg1: memref<4096x2160xf32>, %arg2: memref<4096x2160xf32>, %arg3: memref<4096x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.110209078 : f32
    %cst_0 = arith.constant -0.183681786 : f32
    %cst_1 = arith.constant 0.114441216 : f32
    %cst_2 = arith.constant -0.188681662 : f32
    %cst_3 = arith.constant 0.840896427 : f32
    %cst_4 = arith.constant -0.606530666 : f32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %0, %alloca[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_8[] : memref<f32>
    %alloca_9 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_9[] : memref<f32>
    %alloca_10 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_10[] : memref<f32>
    %alloca_11 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_11[] : memref<f32>
    %alloca_12 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_12[] : memref<f32>
    %alloca_13 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_13[] : memref<f32>
    %alloca_14 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_14[] : memref<f32>
    call @deriche_kernel_0(%alloca_12, %alloca_11, %alloca_14, %arg0, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_1(%alloca_6, %alloca, %alloca_10, %alloca_9, %arg3, %arg0) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_2(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_3(%alloca_13, %alloca_12, %alloca_11, %arg1, %arg2) : (memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_4(%alloca_8, %alloca_7, %alloca_6, %alloca, %arg3, %arg1) : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    call @deriche_kernel_5(%arg2, %arg3, %arg1) : (memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
    return
  }
  func.func private @deriche_kernel_0(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_1(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_2(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_3(memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_4(memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
  func.func private @deriche_kernel_5(memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>)
}

