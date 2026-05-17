target_sources(
    moke_perf_benchmark

    PRIVATE
    matmul_benchmark.cpp
    matmul_baseline.cu
    matmul_mdspan.cu
    matmul_tensor.cu
)
