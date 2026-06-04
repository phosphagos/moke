moke_add_test(
    test_moke_core
    core/test_meta.cpp
    core/test_tensor.cpp
)

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
)
