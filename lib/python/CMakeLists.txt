#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

# Set the installation path for Python modules
set(PYTHON_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin/pypack)

# Define the Python file to be installed
set(ADORA_COMMON_FILE adora_common.py)

# Install the Python module
install(FILES ${ADORA_COMMON_FILE} DESTINATION ${PYTHON_INSTALL_PATH})