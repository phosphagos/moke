moke_add_test(
    test_moke_core
    core/test_meta.cpp
    core/test_tensor.cpp
    core/test_static_layout.cpp
    core/test_static_tensor.cpp
    core/test_static_tensor.device.cpp
)

string(TOUPPER ${MOKE_PLATFORM} MOKE_DEVICE_LANGUAGE)
set_source_files_properties(
    core/test_static_tensor.device.cpp
    PROPERTIES LANGUAGE ${MOKE_DEVICE_LANGUAGE})

moke_add_test(
    test_moke_native
    native/test_memory.cpp
    native/test_algorithm.cpp
    native/test_timer.cpp
)

moke_add_test(
    test_moke_utils
    utils/test_vector.cpp
    utils/test_functional.cpp
    utils/test_profiler.cpp
    utils/test_meta_utils.cpp
)
