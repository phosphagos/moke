target_sources(
    moketest

    PRIVATE
    profiler.hip
)

target_link_libraries(moketest PUBLIC hip::host)
