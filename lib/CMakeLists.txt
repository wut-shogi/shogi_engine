set(cudaFiles
    Bitboard.cu
    Board.cu
    kernels.cu
    MoveGen.cu
)

set(headerFiles
    Bitboard.h
    Board.h
    MoveGen.h
    Rules.h
)

set(sourceFiles
    engine.cpp
)

add_library(shogi_engine STATIC
    ${cudaFiles} ${headerFiles} ${sourceFiles}
)

target_include_directories(shogi_engine PUBLIC ../include)
