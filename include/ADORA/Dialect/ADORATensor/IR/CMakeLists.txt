add_mlir_dialect(ADORATensorOps ADORATensor)
add_mlir_doc(ADORATensorOps ADORATensorOps Dialects/ -gen-op-doc )

set(LLVM_TARGET_DEFINITIONS ADORATensorBase.td)
mlir_tablegen(ADORATensorOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ADORATensorOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRADORATensorOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS ADORATensorOps.td)
mlir_tablegen(ADORATensorOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(ADORATensorOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRADORATensorOpsEnumsGen)

set(LLVM_TARGET_DEFINITIONS ADORATensorOps.td)
mlir_tablegen(ADORATensorOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=ADORATensor)
mlir_tablegen(ADORATensorOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ADORATensor)
add_public_tablegen_target(MLIRADORATensorOpsAttributesIncGen)