cmake_minimum_required(VERSION 3.16)

project(
    MNIST-DNN-Training
    VERSION 1.0
    DESCRIPTION "Training an Deep Neural Network (DNN) on the MNIST dataset."
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(extern/termcolor)
add_subdirectory(extern/spdlog)

add_executable(runme ./src/main.cpp)
target_compile_definitions(spdlog PUBLIC SPDLOG_COMPILED_LIB)
target_link_libraries(runme spdlog::spdlog)