cmake_minimum_required(VERSION 3.16)

project(MNIST-DNN-Training CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(runme ./src/main.cpp)