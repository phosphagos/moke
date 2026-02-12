include(CTest)
enable_testing()
find_package(GTest CONFIG REQUIRED)

set(MOKE_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

macro(moke_add_test test_name)
    add_executable(${test_name} ${ARGN})
    target_link_libraries(${test_name} PRIVATE moke mokeutils)
    target_link_libraries(${test_name} PRIVATE GTest::gtest GTest::gtest_main)
    target_include_directories(${test_name} PRIVATE ${MOKE_TEST_SOURCE_DIR}/include)
    add_test(AllTestInMain ${test_name})
endmacro()

add_subdirectory(unit_test)
add_subdirectory(perf_test)
