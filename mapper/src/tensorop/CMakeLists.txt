set(SPDLOG_NO_EXCEPTIONS ON)

add_library(ADORAMapTensorOp
  tensorop.cpp 
  
  Gemm/mapGemm.cpp
  Gemm/WSGemm.cpp
)

target_link_libraries(ADORAMapTensorOp
  ADORATensorOps
  ADORATensorInterface
)