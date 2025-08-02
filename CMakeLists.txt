cmake_minimum_required(VERSION 3.21.0)
project(moke VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)
if (DEFINED MOKE_ENABLE_TEST)
    add_subdirectory(test)
endif()
