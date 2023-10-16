add_executable(engine_tests
    example_test.cpp
    main.cpp)

target_link_libraries(engine_tests
    PRIVATE gtest
    PRIVATE shogi_engine)

# Run tests after build
add_custom_command(TARGET engine_tests POST_BUILD
    COMMAND $<TARGET_FILE:engine_tests>)