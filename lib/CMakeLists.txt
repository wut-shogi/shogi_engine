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

set(dllFiles
    include/shogi_engine.h
    engine/shogi_engine.cpp
)

add_library(shogi_engine STATIC
    ${cudaFiles} ${headerFiles} ${sourceFiles}
)

add_library(shogi_engine_lib SHARED 
    ${cudaFiles} ${headerFiles} ${sourceFiles} ${dllFiles}
)

target_include_directories(shogi_engine PUBLIC ../include)

target_include_directories(shogi_engine_lib PUBLIC include)
