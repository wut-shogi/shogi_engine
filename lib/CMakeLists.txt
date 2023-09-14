add_library(shogi_engine
    engine.cpp
    kernels.cu
)

target_include_directories(shogi_engine PUBLIC ../include)
