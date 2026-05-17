target_sources(
    moke_perf_benchmark

    PRIVATE
    matmul_benchmark.cpp
    matmul_baseline.hip
    matmul_tensor.hip
)
