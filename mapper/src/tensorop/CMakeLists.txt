set(SPDLOG_NO_EXCEPTIONS ON)

add_library(ADORAMapTensorOp
  tensorop.cpp 
  
  ## Gemm op
  mapGemm.cpp
)

target_link_libraries(ADORAMapTensorOp
  ADORATensorOps
  ADORATensorInterface
  ADORATensorOpGemmLower
)
# add_subdirectory(Gemm)