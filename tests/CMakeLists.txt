add_executable(engine_tests
    bitboardTests.cpp
    moveGenTests.cpp
    example_test.cpp
    main.cpp)

target_link_libraries(engine_tests
    PRIVATE gtest
    PRIVATE shogi_engine_core)

target_include_directories(engine_tests PRIVATE ../core)
# Run tests after build
add_custom_command(TARGET engine_tests POST_BUILD
    COMMAND $<TARGET_FILE:engine_tests>)