cmake_minimum_required(VERSION 3.16)

project(
    MNIST-DNN-Training
    VERSION 1.0
    DESCRIPTION "Training an Deep Neural Network (DNN) on the MNIST dataset."
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(extern/glog)

include_directories(extern/termcolor)

# glog library for logging
add_executable(runme ./src/main.cpp)
target_link_libraries(runme glog::glog)