add_library(moke "")
add_subdirectory(native/${MOKE_PLATFORM})

target_include_directories(moke PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(moke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
