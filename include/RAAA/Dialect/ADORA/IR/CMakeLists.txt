add_mlir_dialect(ADORAOps ADORA)
add_mlir_doc(ADORAOps ADORAOps Dialects/ -gen-op-doc )

set(LLVM_TARGET_DEFINITIONS ADORABase.td)
mlir_tablegen(ADORAOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ADORAOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRADORAOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS ADORAOps.td)
mlir_tablegen(ADORAOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(ADORAOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRADORAOpsEnumsGen)

set(LLVM_TARGET_DEFINITIONS ADORAOps.td)
mlir_tablegen(ADORAOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=ADORA)
mlir_tablegen(ADORAOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ADORA)
add_public_tablegen_target(MLIRADORAOpsAttributesIncGen)