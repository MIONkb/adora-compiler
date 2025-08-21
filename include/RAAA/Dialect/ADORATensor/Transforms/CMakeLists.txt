set(LLVM_TARGET_DEFINITIONS Passes.td)

mlir_tablegen(Passes.h.inc -gen-pass-decls -name ADORATensor)
# mlir_tablegen(Passes.cpp.inc -gen-pass-defs -name ADORATensor)

add_public_tablegen_target(ADORATensorPassIncGen)

add_mlir_doc(Passes ADORATensorPasses ./ -gen-pass-doc)