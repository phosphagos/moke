add_library(moketest "")
add_subdirectory(src/common)
add_subdirectory(src/${MOKE_PLATFORM})

target_link_libraries(moketest PUBLIC moke)
target_include_directories(moketest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(moketest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
