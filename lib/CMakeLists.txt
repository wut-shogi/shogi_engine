set(cudaFiles
    engine/kernels.cu
)

set(headerFiles
    engine/Bitboard.h
    engine/Board.h
    engine/BoardVisualization.h
    engine/MoveGen.h
    engine/MoveGenHelpers.h
    engine/Rules.h
    engine/Square.h
)

set(sourceFiles
    engine/Bitboard.cpp
    engine/Board.cpp
    engine/engine.cpp
    engine/MoveGen.cpp
    engine/MoveGenHelpers.cpp
)

add_library(shogi_engine STATIC
    ${cudaFiles} ${headerFiles} ${sourceFiles}
)

target_include_directories(shogi_engine PUBLIC ../include)
