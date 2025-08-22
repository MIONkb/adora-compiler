add_mlir_doc(LowerPasses ADORALowerPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS LowerPasses.td)
mlir_tablegen(LowerPasses.h.inc -gen-pass-decls -name ADORALower)
add_public_tablegen_target(ADORALowerPassIncGen)

