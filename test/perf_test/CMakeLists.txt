moke_add_test(
    moke_perf_benchmark
    memory_benchmark.cpp
    random_benchmark.cpp
)

if (MOKE_PLATFORM STREQUAL cuda)
    add_subdirectory(tensor_benchmark)
endif ()