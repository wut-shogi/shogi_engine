set(headerFiles
    engine/bitboard.h
    engine/board.h
    engine/cpuInterface.h
    include/engine.h
    engine/game.h
    engine/gameTree.h
    engine/gpuInterface.h
    engine/lookUpTables.h
    engine/moveGen.h
    engine/moveGenHelpers.h
    engine/rules.h
    engine/square.h
)

set(sourceFiles
    engine/bitboard.cu
    engine/board.cu
    engine/cpuInterface.cu
    engine/engine.cu
    engine/game.cu
    engine/moveGen.cu
    engine/moveGenHelpers.cu
    engine/gpuInterface.cu
    engine/lookUpTables.cu
)

set(dllFiles
    ../include/shogi_engine.h
    engine/shogi_engine.cu
)

add_library(shogi_engine_core STATIC
    ${headerFiles} ${sourceFiles}
)
target_include_directories(shogi_engine_core PUBLIC include)
set_target_properties(shogi_engine_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(shogi_engine_core PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(shogi_engine SHARED ${dllFiles})
target_link_libraries(shogi_engine PRIVATE shogi_engine_core)
target_include_directories(shogi_engine PRIVATE ../include)
