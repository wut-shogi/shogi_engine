add_library(shogi_engine
    instance.cpp
    interface.cpp
    kernels.cu
)

target_include_directories(shogi_engine PUBLIC ../include)
