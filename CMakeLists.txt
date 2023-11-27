cmake_minimum_required(VERSION 3.22)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

project(
    shogi_engine
    VERSION 0.1
    DESCRIPTION "TODO"
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(${CMAKE_BINARY_DIR})

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

add_subdirectory(lib)
add_subdirectory(app)


add_subdirectory(external/gtest EXCLUDE_FROM_ALL)
add_subdirectory(tests)
