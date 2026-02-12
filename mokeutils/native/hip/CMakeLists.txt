target_sources(
    mokeutils

    PRIVATE
    memory.cpp
    timer.cpp
    kernels/memory.hip
    kernels/compare.hip
    kernels/random_fill.hip
)
