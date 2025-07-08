add_library(moke "")
add_subdirectory(common)
target_include_directories(moke PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(moke PRIVATE ${CMAKE_SOURCE_DIR})
