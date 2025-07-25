add_library(moketest "")
add_subdirectory(moketest/common)
add_subdirectory(moketest/${MOKE_PLATFORM})

target_link_libraries(moketest PUBLIC moke)
target_include_directories(moketest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(moketest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/moketest)

add_subdirectory(zoo)
