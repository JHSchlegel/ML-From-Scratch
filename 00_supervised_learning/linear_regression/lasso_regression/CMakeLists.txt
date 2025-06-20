cmake_minimum_required(VERSION 3.10)
project(LassoRegression)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3 package
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add executable
add_executable(lasso_reg_project 
    main.cpp
    lasso_reg.cpp
    lasso_reg.hpp
)

# Link Eigen3
target_link_libraries(lasso_reg_project Eigen3::Eigen)

# Set output directory
set_target_properties(lasso_reg_project PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lasso_reg_project PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(lasso_reg_project PRIVATE /W4)
endif()

# Optional: Add debug symbols in Debug mode
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Optional: Enable optimization in Release mode
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")