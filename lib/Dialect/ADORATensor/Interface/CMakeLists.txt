add_mlir_library(ADORATensorInterface
  SystolicImplInterface.cpp
  ShapeInferenceOpInterface.cpp
  # CloneFromOtherDialectOpInterface.cpp

  DEPENDS
  # ADORATensorCloneFromOtherDialectOpInterfaceIncGen
  ADORATensorShapeInferenceOpInterfaceIncGen
  ADORATensorSystolicImplInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  LLVMSupport
  )
