moke_add_test(
    moke_tensor_perf_benchmark

    matmul_benchmark.cpp
    matmul_baseline.hip
    matmul_tensor.hip
)
