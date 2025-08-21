add_mlir_library(ADORATensorInterface
  ShapeInferenceOpInterface.cpp
  # CloneFromOtherDialectOpInterface.cpp

  DEPENDS
  # ADORATensorCloneFromOtherDialectOpInterfaceIncGen
  ADORATensorShapeInferenceOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  LLVMSupport
  )
