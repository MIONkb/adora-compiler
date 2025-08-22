set(SPDLOG_NO_EXCEPTIONS ON)

add_library(ADORAMapTensorOp
  tensorop.cpp 
  mapGemm.cpp 
)

target_link_libraries(ADORAMapTensorOp
  ADORATensorOps
  ADORATensorInterface
)