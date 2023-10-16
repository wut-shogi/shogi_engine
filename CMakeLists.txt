cmake_minimum_required(VERSION 3.22)

project(
    shogi_engine
    VERSION 0.1
    DESCRIPTION "TODO"
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

add_subdirectory(lib)
add_subdirectory(app)
add_subdirectory(tests)

add_subdirectory(external/gtest EXCLUDE_FROM_ALL)
