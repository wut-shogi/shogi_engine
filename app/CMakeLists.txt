add_executable(shogi_main main.cpp)

target_link_libraries(shogi_main PRIVATE shogi_engine)
target_include_directories(shogi_main PRIVATE ../include)
