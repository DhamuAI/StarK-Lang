add_mlir_dialect_library(MLIRStarK
                        StarKDialect.cpp
                        StarKOps.cpp
                        LowerToAffine.cpp
                        LowerToLLVM.cpp

        
        
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/StarK 

        DEPENDS
        MLIRStarKOpsIncGen  

        LINK_LIBS PUBLIC  
        MLIRIR  
        )