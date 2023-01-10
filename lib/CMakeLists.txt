add_mlir_dialect_library(MLIRStarK
                        StarKDialect.cpp
                        StarKOps.cpp
                        LowerToAffine.cpp
                        LowerToLLVM.cpp
                        ShapeInferenceOpInterface.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/StarK 

        DEPENDS
        MLIRStarKOpsIncGen
        MLIRStarKShapeInferenceOpInterfaceIncGen

        LINK_LIBS PUBLIC  
        MLIRIR  
        )


