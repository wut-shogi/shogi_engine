add_library(shogi_engine STATIC SHARED
    engine.cpp
    kernels.cu
)

target_include_directories(shogi_engine PUBLIC ../include)
