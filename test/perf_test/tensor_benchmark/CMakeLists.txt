moke_add_test(
    moke_tensor_perf_benchmark

    matmul_benchmark.cpp
    matmul_baseline.cu
    matmul_mdspan.cu
    matmul_tensor.cu
)
