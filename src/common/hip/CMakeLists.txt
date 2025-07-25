target_sources(
    moke

    PRIVATE
    memory.hip
    runtime.hip
)

target_link_libraries(moke PUBLIC hip::host)
