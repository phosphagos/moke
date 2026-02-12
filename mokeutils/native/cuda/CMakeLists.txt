target_sources(
    mokeutils

    PRIVATE
    memory.cpp
    status.cpp
    kernels/memory.cu
    kernels/compare.cu
    kernels/random_fill.cu
)

target_link_libraries(mokeutils PUBLIC CUDA::cuda_driver CUDA::curand)
