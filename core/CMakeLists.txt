set(headerFiles
    engine/Bitboard.h
    engine/Board.h
    engine/CPUsearchHelpers.h
    include/engine.h
    engine/evaluation.h
    engine/game.h
    engine/GPUsearchHelpers.h
    engine/lookUpTables.h
    engine/MoveGen.h
    engine/MoveGenHelpers.h
    engine/Rules.h
    engine/search.h
    engine/Square.h
    engine/USIconverter.h
)

set(sourceFiles
    engine/Bitboard.cu
    engine/Board.cu
    engine/CPUsearchHelpers.cu
    engine/engine.cu
    engine/evaluation.cu
    engine/game.cu
    engine/MoveGen.cu
    engine/MoveGenHelpers.cu
    engine/GPUsearchHelpers.cu
    engine/lookUpTables.cu
    engine/search.cu
    engine/USIconverter.cu
)

set(dllFiles
    ../include/shogi_engine.h
    engine/shogi_engine.cpp
)

if(NOT SHOGI_CUDA_SUPPORT)
    set_source_files_properties(${sourceFiles} PROPERTIES LANGUAGE CXX)
endif()

add_library(shogi_engine_core SHARED ${headerFiles} ${sourceFiles})
add_library(shogi_engine_core_static STATIC ${headerFiles} ${sourceFiles})

target_include_directories(shogi_engine_core PUBLIC include)
target_include_directories(shogi_engine_core_static PUBLIC include)

if(SHOGI_CUDA_SUPPORT)
    set_target_properties(shogi_engine_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(shogi_engine_core PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    set_target_properties(shogi_engine_core_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(shogi_engine_core_static PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

add_library(shogi_engine SHARED ${dllFiles})
target_link_libraries(shogi_engine PRIVATE shogi_engine_core)
target_include_directories(shogi_engine PRIVATE ../include)
