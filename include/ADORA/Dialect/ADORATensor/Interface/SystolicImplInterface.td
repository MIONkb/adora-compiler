// SPDX-License-Identifier: Apache-2.0

//===- ShapeInferenceInterface.td - ShapeInference Interface -*- tablegen -===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Defines the operations of the Shape Inference Op Interface.
//
//===----------------------------------------------------------------------===//

#ifdef ADORATENSOR_SYSTOLIC_IMPLEMENT_INTERFACE_TD
#else
#define ADORATENSOR_SYSTOLIC_IMPLEMENT_INTERFACE_TD

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def SystolicImplInterface : OpInterface<"SystolicImplInterface"> {
  let description = [{
    
  }];

  let methods = [
  /* stationary kind interface */
InterfaceMethod<
  /* doc string */
  "Return the attribute key used to store the stationary kind.",
  /* return type */
  "mlir::StringRef",
  /* method name */
  "getStationaryKindAttrStr",
  /* arguments */
  (ins),
  /* defaultImplementation= */
  [{
  return "stationary_kind";
  }]
>,

InterfaceMethod<
  /* doc string */
  "Set the stationary kind attribute for this operation.",
  /* return type */
  "void",
  /* method name */
  "setStationaryKind",
  /* arguments */
  (ins "MatMulStrategy":$method),
  /* defaultImplementation= */
[{
  StringAttr methodAttr = StringAttr::get(
      tablegen_opaque_val->getContext(), getMethodStrRef(method)
    );
  tablegen_opaque_val->setAttr(
      getStationaryKindAttrStr(impl, tablegen_opaque_val), 
      methodAttr
    );
}]
>,

InterfaceMethod<
  /* doc string */
  "Get the stationary kind attribute value of this operation.",
  /* return type */
  "mlir::StringRef",
  /* method name */
  "getStationaryKind",
  /* arguments */
  (ins),
  /* defaultImplementation= */
[{
  StringRef methodAttr = tablegen_opaque_val->getAttr(
      getStationaryKindAttrStr(impl, tablegen_opaque_val)
    ).cast<StringAttr>().strref();
  return methodAttr;
}]
>,

  /* matmul tile size interface */
InterfaceMethod<
  /* doc string */
  "Return the attribute key used to store the matmul tile size.",
  /* return type */
  "mlir::StringRef",
  /* method name */
  "getTileSizeAttrStr",
  /* arguments */
  (ins),
  /* defaultImplementation= */
  [{
  return "tile_size";
  }]
>,

InterfaceMethod<
  /* doc string */
  "Set the tile size attribute for this operation.",
  /* return type */
  "void",
  /* method name */
  "setTileSize",
  /* arguments */
  (ins "ArrayRef<int64_t>":$tileSize),
  /* defaultImplementation= */
[{
  DenseI64ArrayAttr TileAttr = DenseI64ArrayAttr::get(
      tablegen_opaque_val->getContext(), tileSize
    );
  tablegen_opaque_val->setAttr(
      getTileSizeAttrStr(impl, tablegen_opaque_val), 
      TileAttr
    );
}]
>,

InterfaceMethod<
  /* doc string */
  "Get the stationary kind attribute value of this operation.",
  /* return type */
  "ArrayRef<int64_t>",
  /* method name */
  "getTileSize",
  /* arguments */
  (ins),
  /* defaultImplementation= */
[{
  DenseI64ArrayAttr arrayattr = cast<DenseI64ArrayAttr>(
      tablegen_opaque_val->getAttr(
      getTileSizeAttrStr(impl, tablegen_opaque_val)
    ));
  
  return arrayattr.asArrayRef();
}]
>  

  ];
  
  let cppNamespace = "::mlir::ADORA::ADORATensor";
}


#endif // SHAPE_INFERENCE_OP_INTERFACE
