add_library(mokeutils "")
add_subdirectory(native/${MOKE_PLATFORM})

target_link_libraries(mokeutils PUBLIC moke)
target_include_directories(mokeutils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(mokeutils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
