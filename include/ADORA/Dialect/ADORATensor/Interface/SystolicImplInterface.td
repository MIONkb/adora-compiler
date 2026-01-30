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

def SystolicImplInterface : OpInterface<"SystolicImplInterface">
{
  let description = [{
    
  }];

  let methods = [
      // Stationary Kind (Dataflow)
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
  }]>,

      InterfaceMethod<
          /* doc string */
          "Set the stationary kind attribute for this operation.",
          /* return type */
          "void",
          /* method name */
          "setStationaryKind",
          /* arguments */
          (ins "DataflowStrategy":$method),
          /* defaultImplementation= */
          [{
  StringAttr methodAttr = StringAttr::get(
      tablegen_opaque_val->getContext(), getDataflowStrategyStrRef(method)
    );
  tablegen_opaque_val->setAttr(
      getStationaryKindAttrStr(impl, tablegen_opaque_val), 
      methodAttr
    );
  }]>,

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
  }]>,

      // Tile Size
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
  }]>,

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
  }]>,

      InterfaceMethod<
          /* doc string */
          "Get the tile size attribute value of this operation.",
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
  }]>,

      // Algorithm
      InterfaceMethod<
          "Return the attribute key used to store the compute algorithm.",
          "mlir::StringRef", "getAlgorithmAttrStr", (ins),
          [{ return "algorithm"; }]>,
      InterfaceMethod<
          "Set the algorithm attribute.",
          "void", "setAlgorithm", (ins "ComputeAlgorithm":$algo),
          [{
    StringAttr algoAttr = StringAttr::get(tablegen_opaque_val->getContext(), getComputeAlgorithmStrRef(algo));
    tablegen_opaque_val->setAttr(getAlgorithmAttrStr(impl, tablegen_opaque_val), algoAttr);
  }]>,
      InterfaceMethod<
          "Get the algorithm attribute value.",
          "mlir::StringRef", "getAlgorithm", (ins),
          [{
    auto attr = tablegen_opaque_val->getAttr(getAlgorithmAttrStr(impl, tablegen_opaque_val));
    if (!attr) return ""; 
    return attr.cast<StringAttr>().strref();
  }]>,

      // Loop Order
      InterfaceMethod<
          "Return the attribute key used to store the loop order.",
          "mlir::StringRef", "getLoopOrderAttrStr", (ins),
          [{ return "loop_order"; }]>,
      InterfaceMethod<
          "Set the loop order attribute.",
          "void", "setLoopOrder", (ins "ArrayRef<int64_t>":$order),
          [{
    DenseI64ArrayAttr orderAttr = DenseI64ArrayAttr::get(tablegen_opaque_val->getContext(), order);
    tablegen_opaque_val->setAttr(getLoopOrderAttrStr(impl, tablegen_opaque_val), orderAttr);
  }]>,
      InterfaceMethod<
          "Get the loop order attribute value.",
          "ArrayRef<int64_t>", "getLoopOrder", (ins),
          [{
    auto attr = tablegen_opaque_val->getAttr(getLoopOrderAttrStr(impl, tablegen_opaque_val));
    if (!attr) return {};
    return attr.cast<DenseI64ArrayAttr>().asArrayRef();
  }]>

  ];

  let cppNamespace = "::mlir::ADORA::ADORATensor";
}

#endif // SHAPE_INFERENCE_OP_INTERFACE
