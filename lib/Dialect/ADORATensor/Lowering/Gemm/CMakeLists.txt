set(SPDLOG_NO_EXCEPTIONS ON)

add_library(ADORATensorOpGemmLower
  WSGemm.cpp
  ISGemm.cpp
  OSGemm.cpp
  NestedLoopGeneration.cpp
)

target_link_libraries(ADORATensorOpGemmLower
  ADORATensorOps
  ADORATensorInterface
  MLIRADORAKernelExtract
  MLIRADORATransformsUtility
)