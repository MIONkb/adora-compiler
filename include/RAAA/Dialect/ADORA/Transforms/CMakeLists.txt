add_mlir_doc(Passes ADORAPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ADORA)
add_public_tablegen_target(ADORAPassIncGen)

