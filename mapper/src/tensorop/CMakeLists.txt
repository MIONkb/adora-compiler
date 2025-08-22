set(SPDLOG_NO_EXCEPTIONS ON)

add_library(ADORAMapTensorOp
  tensorop.cpp 
  mapGemm.cpp 
)

add_dependencies(ADORAMapTensorOp ADORATensorOps)