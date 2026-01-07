set(ADORA_TEST_DEPENDS
  cgra-opt
  FileCheck
)

set(ADORA_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ADORA_TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(ADORA_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

configure_lit_site_cfg(
  ${ADORA_TEST_DIR}/lit.site.cfg.py.in
  ${ADORA_TEST_OUTPUT_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${ADORA_TEST_DIR}/lit.cfg.py
)

add_lit_testsuite(check-adora "Running the ADORA regression tests"
  ${ADORA_TEST_OUTPUT_DIR}
  DEPENDS ${ADORA_TEST_DEPENDS}
)

add_lit_testsuites(ADORA ${ADORA_TEST_DIR}
  DEPENDS ${ADORA_TEST_DEPENDS}
)
