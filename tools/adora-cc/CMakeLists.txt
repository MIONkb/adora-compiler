#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

set(PYTHON_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin)

set(ADORACC_SUBFILES
  __init__.py
  main.py)

set(ADORACC_FILES
  adoracc.py
  adoracc/__init__.py
  adoracc/main.py)

set(ADORACC_TARGETS ${ADORACC_FILES})
list(TRANSFORM ADORACC_TARGETS PREPEND ${PROJECT_BINARY_DIR}/bin/)


#if(${AIE_LINKER} STREQUAL NONE)
#set(CONFIG_DISABLE_LINK True)
#else()
#set(CONFIG_DISABLE_LINK False)
#endif()


# configure_file(adoracc/configure.py.in ${PROJECT_BINARY_DIR}/bin/adoracc/configure.py)
install(PROGRAMS ${PROJECT_BINARY_DIR}/bin/adoracc/configure.py DESTINATION ${PYTHON_INSTALL_PATH}/adoracc)

# Stuff our python into the build area:
add_custom_target(adoracc.py ALL DEPENDS ${ADORACC_TARGETS})

# This chicanery is necessary to ensure executable permissions.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/copy_adoracc.cmake"
"file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/adoracc.py
DESTINATION ${PROJECT_BINARY_DIR}/bin
FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_WRITE
GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)")

add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bin/adoracc.py COMMAND
${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/copy_adoracc.cmake)

foreach(file ${ADORACC_SUBFILES})
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bin/adoracc/${file}
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/adoracc/${file}
                   ${PROJECT_BINARY_DIR}/bin/adoracc/${file}
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/adoracc/${file})
endforeach()

# Install our python stuff too.
install(PROGRAMS adoracc.py DESTINATION ${PYTHON_INSTALL_PATH})
foreach(file ${ADORACC_SUBFILES})
  install(PROGRAMS adoracc/${file} DESTINATION ${PYTHON_INSTALL_PATH}/adoracc)
endforeach()
