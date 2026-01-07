import os

import lit.formats
import lit.util

from lit.llvm import llvm_config

config.name = "ADORA"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.adora_test_output_dir, "test")

llvm_config.use_default_substitutions()

config.substitutions.append(("%cgra-opt", os.path.join(config.adora_tools_dir, "cgra-opt")))
config.substitutions.append(("%FileCheck", os.path.join(config.llvm_tools_dir, "FileCheck")))

config.excludes = [
    "CMakeLists.txt",
    "lit.site.cfg.py.in",
]
