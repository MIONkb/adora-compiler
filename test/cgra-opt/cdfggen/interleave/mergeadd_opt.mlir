module attributes {} {
  func.func @kernel_merge4(%arg0: memref<?xi16>, %arg1: memref<?xi16>, %arg2: memref<?xi16>, %arg3: memref<?xi16>, %arg4: memref<?xi16>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = ADORA.BlockLoad %arg0 [0] : memref<?xi16> -> memref<4xi16>  {Id = "0", KernelName = "merge"}
    %1 = ADORA.BlockLoad %arg1 [0] : memref<?xi16> -> memref<4xi16>  {Id = "1", KernelName = "merge"}
    %2 = ADORA.BlockLoad %arg2 [0] : memref<?xi16> -> memref<4xi16>  {Id = "2", KernelName = "merge"}
    %3 = ADORA.BlockLoad %arg3 [0] : memref<?xi16> -> memref<4xi16>  {Id = "3", KernelName = "merge"}
    %b = ADORA.BlockLoad %arg3 [0] : memref<?xi16> -> memref<4xi16>  {Id = "5", KernelName = "merge"}
    %4 = ADORA.LocalMemAlloc memref<16xi16>  {Id = "4", KernelName = "merge"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 4 {
        %5 = affine.load %0[%arg5] : memref<4xi16>
        %6 = affine.load %1[%arg5] : memref<4xi16>
        %7 = affine.load %2[%arg5] : memref<4xi16>
        %8 = affine.load %3[%arg5] : memref<4xi16>
        affine.for %arg6 = 0 to 4 {

        }
        %9 = ADORA.interleaver %5, %6, %7, %8 : i16, i16, i16, i16 -> vector<4xi16>
        %a = affine.vector_load %b[0] : memref<4xi16>, vector<4xi16>
        %add = arith.addi %9, %a : vector<4xi16>
        affine.vector_store %add, %4[%arg5 * 4] : memref<16xi16>, vector<4xi16>
        // affine.for %arg6 = 0 to 4 {
        //   %10 = vector.extract %9[%arg6] : vector<4xi16>
        //   affine.store %10, %4[%arg5 * 4 + %arg6] : memref<16xi16>
        // }
      } 
      ADORA.terminator
    }{KernelName = "merge"}
    ADORA.BlockStore %4, %arg4 [0] : memref<16xi16> -> memref<?xi16>  {Id = "4", KernelName = "merge"}
    return
  }
}

