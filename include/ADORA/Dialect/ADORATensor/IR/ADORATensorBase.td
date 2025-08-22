//===-- ADORATensor_BASE.td - Test dialect definitions ---------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines the ADORATensor dialect
// 
//===----------------------------------------------------------------------===//

#ifndef ADORATensor_BASE
#define ADORATensor_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// ADORATensor Dialect.
//===----------------------------------------------------------------------===//

def ADORATensor_Dialect : Dialect {
  let name = "ADORATensor";
  let cppNamespace = "::mlir::ADORA::ADORATensor";

  let description = [{
    "a sub-dialect for tensor in ADORA"
  }];

  let dependentDialects = [
    // "scf::SCFDialect",
    // "::mlir::func"
    // "::mlir::ADORA"
  ];

  // let hasOperationAttrVerify = 1;
}

#endif // ADORATensor_BASE
