add_mlir_dialect_library(MLIRADORAMisc
    MiscPasses.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/

    DEPENDS
    MLIRADORAMiscPassesIncGen

	LINK_LIBS PUBLIC
	MLIRIR
    MLIRPass
	)
