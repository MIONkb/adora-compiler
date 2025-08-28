set(SPDLOG_NO_EXCEPTIONS ON)

add_library(ADORATensorOpGemmLower
  WSGemm.cpp
)

target_link_libraries(ADORATensorOpGemmLower
  ADORATensorOps
  ADORATensorInterface
  MLIRADORAKernelExtract
)