target_sources(
    moketest

    PRIVATE
    device_profiler.cu
)

target_link_libraries(moketest PUBLIC CUDA::cudart)
