moke_add_test(
    moke_perf_benchmark
    memory_benchmark.cpp
    random_benchmark.cpp
)

add_subdirectory(tensor_benchmark/${MOKE_PLATFORM})
