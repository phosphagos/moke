target_sources(
    moke

    PRIVATE
    memory.cu
    runtime.cu
)

target_link_libraries(moke PUBLIC CUDA::cudart)
