# This file allows users to call find_package(ADORA) and pick up our targets.


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was ADORAConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../" ABSOLUTE)

####################################################################################

include(CMakeFindDependencyMacro)

find_dependency(MLIR REQUIRED CONFIG)

include("${CMAKE_CURRENT_LIST_DIR}/ADORATargets.cmake")
