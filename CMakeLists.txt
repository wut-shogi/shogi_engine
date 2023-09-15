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

add_subdirectory(lib)
add_subdirectory(app)
