cmake_minimum_required(VERSION 3.10)

project(random_forest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add source files
add_executable(random_forest_app
    main.cpp
    decision_tree.cpp
    random_forest.cpp
)

# Enable warnings
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANG)
  target_compile_options(random_forest_app PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS random_forest_app DESTINATION bin)

enable_testing()
# Simple test to see if it runs
add_test(NAME random_forest_run COMMAND random_forest_app) 