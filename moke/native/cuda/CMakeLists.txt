target_sources(
    moke

    PRIVATE
    runtime.cpp
)

target_compile_definitions(moke PUBLIC -DMOKE_PLATFORM_CUDA)
target_link_libraries(moke PUBLIC CUDA::cudart)
