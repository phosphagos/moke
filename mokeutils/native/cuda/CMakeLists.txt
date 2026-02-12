target_sources(
    mokeutils

    PRIVATE
    memory.cpp
    memory.cu
)

target_link_libraries(mokeutils PUBLIC CUDA::cuda_driver)
