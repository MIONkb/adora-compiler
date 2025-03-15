#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cholesky(%arg0: memref<?x2000xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg1 = 0 to 2000 {
      affine.for %arg2 = 0 to #map(%arg1) {

        ADORA.kernel{
        affine.for %arg3 = 0 to #map(%arg2) {
          %5 = affine.load %arg0[%arg1, %arg3] : memref<?x2000xf32>
          %6 = affine.load %arg0[%arg2, %arg3] : memref<?x2000xf32>
          %7 = arith.mulf %5, %6 : f32
          %8 = affine.load %arg0[%arg1, %arg2] : memref<?x2000xf32>
          %9 = arith.subf %8, %7 : f32
          affine.store %9, %arg0[%arg1, %arg2] : memref<?x2000xf32>
        }
        ADORA.terminator}
        

        %2 = affine.load %arg0[%arg2, %arg2] : memref<?x2000xf32>
        %3 = affine.load %arg0[%arg1, %arg2] : memref<?x2000xf32>
        %4 = arith.divf %3, %2 : f32
        affine.store %4, %arg0[%arg1, %arg2] : memref<?x2000xf32>
      }

      ADORA.kernel{
      affine.for %arg2 = 0 to #map(%arg1) {
        %2 = affine.load %arg0[%arg1, %arg2] : memref<?x2000xf32>
        %3 = arith.mulf %2, %2 : f32
        %4 = affine.load %arg0[%arg1, %arg1] : memref<?x2000xf32>
        %5 = arith.subf %4, %3 : f32
        affine.store %5, %arg0[%arg1, %arg1] : memref<?x2000xf32>
      }
      ADORA.terminator}

      %0 = affine.load %arg0[%arg1, %arg1] : memref<?x2000xf32>
      %1 = math.sqrt %0 : f32
      affine.store %1, %arg0[%arg1, %arg1] : memref<?x2000xf32>
    }
    return
  }
}

