target_sources(
    moketest

    PRIVATE
    device_profiler.hip
)

target_link_libraries(moketest PUBLIC hip::host)
