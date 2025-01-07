cmake_minimum_required(VERSION 3.15)

project(myLeetcode LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_SOURCE_DIR}/includes)

add_subdirectory(1_two_sum)