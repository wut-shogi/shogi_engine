cmake_minimum_required(VERSION 3.22)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# project description
set(SHOGI_VERSION 0.1)
set(SHOGI_DESCRIPTION "TODO")

# options
option(SHOGI_CUDA_SUPPORT "Enable CUDA compilation and execution." ON)

# handle language settings regarding support for CUDA
if(SHOGI_CUDA_SUPPORT)
    project(
        shogi_engine
        VERSION ${SHOGI_VERSION}
        DESCRIPTION ${SHOGI_DESCRIPTION}
        LANGUAGES CXX CUDA)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    project(
        shogi_engine
        VERSION ${SHOGI_VERSION}
        DESCRIPTION ${SHOGI_DESCRIPTION}
        LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

include_directories(${CMAKE_BINARY_DIR})

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

add_subdirectory(core)
add_subdirectory(app)

add_subdirectory(external/gtest EXCLUDE_FROM_ALL)
add_subdirectory(tests)
