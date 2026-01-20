# lib/Dialect/ADORATensor/Lowering/CONV/CMakeLists.txt

set(SPDLOG_NO_EXCEPTIONS ON)

add_library(ADORATensorOpConvLower
  GenericDirectConv.cpp
)

target_link_libraries(ADORATensorOpConvLower
  ADORATensorOps
  ADORATensorInterface
  MLIRADORAKernelExtract
  MLIRADORATransformsUtility
  ADORATensorOpGemmLower 
)