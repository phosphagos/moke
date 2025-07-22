add_library(moke "")
add_subdirectory(${MOKE_PLATFORM})
target_include_directories(moke PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(moke PRIVATE ${CMAKE_SOURCE_DIR})
