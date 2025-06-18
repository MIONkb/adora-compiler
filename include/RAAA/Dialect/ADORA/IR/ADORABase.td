//===-- ADORA_BASE.td - Test dialect definitions ---------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines the ADORA dialect
// 
//===----------------------------------------------------------------------===//

#ifndef ADORA_BASE
#define ADORA_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// ADORA Dialect.
//===----------------------------------------------------------------------===//

def ADORA_Dialect : Dialect {
  let name = "ADORA";
  let cppNamespace = "::mlir::ADORA";

  let description = [{
    "a dialect for cgra compiler-ADORA"
  }];

  let dependentDialects = [
    // "scf::SCFDialect",
    // "::mlir::func"
  ];

  // let hasOperationAttrVerify = 1;
}

#endif // ADORA_BASE
