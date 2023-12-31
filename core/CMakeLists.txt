set(headerFiles
    engine/ai.h
    engine/Bitboard.h
    engine/Board.h
    engine/BoardVisualization.h
    include/engine.h
    engine/game.h
    engine/MoveGen.h
    engine/MoveGenHelpers.h
    engine/Rules.h
    engine/Square.h
)

set(sourceFiles
    engine/ai.cpp
    engine/Bitboard.cpp
    engine/Board.cpp
    engine/engine.cpp
    engine/game.cpp
    engine/MoveGen.cpp
    engine/MoveGenHelpers.cpp

)

set(dllFiles
    ../include/shogi_engine2.h
    engine/shogi_engine2.cpp
)

add_library(shogi_engine_core SHARED
    ${headerFiles} ${sourceFiles}
)
target_include_directories(shogi_engine_core PUBLIC include)

add_library(shogi_engine SHARED ${dllFiles})
target_link_libraries(shogi_engine PRIVATE shogi_engine_core)
target_include_directories(shogi_engine PRIVATE ../include)
