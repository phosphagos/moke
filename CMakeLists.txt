cmake_minimum_required(VERSION 3.22.0)
project(moke VERSION 0.1.0)

set(MOKE_PLATFORM cuda CACHE STRING "target platform.")
option(MOKE_ENABLE_TESTS "enable building unit tests" OFF)

include(cmake/cpp.cmake)
include(cmake/${MOKE_PLATFORM}.cmake)

add_subdirectory(moke)
add_subdirectory(mokeutils)

if(MOKE_ENABLE_TESTS)
    add_subdirectory(test)
endif()
