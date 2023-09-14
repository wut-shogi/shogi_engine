cmake_minimum_required(VERSION 3.22)

project(
    shogi_engine
    VERSION 0.1
    DESCRIPTION "TODO"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

add_subdirectory(lib)
add_subdirectory(app)
