message("[MOKE] test enabled")
message("[MOKE] test platform: ${MOKE_PLATFORM}")

add_library(moketest "")
add_subdirectory(moketest)
target_link_libraries(moketest PUBLIC moke)
target_include_directories(moke PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(moke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/moketest)
