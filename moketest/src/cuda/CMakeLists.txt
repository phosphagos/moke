target_sources(
    moketest

    PRIVATE
    profiler.cu
)

target_link_libraries(moketest PUBLIC CUDA::cudart)
