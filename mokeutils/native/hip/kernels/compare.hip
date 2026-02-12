#include "moke/native.hpp"
#include "moke/runtime.hpp"
#include "moke/type_traits.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

namespace moke {
template <typename T, auto CompareFunc>
MOKE_KERNEL void compare_all_close_kernel(int *equal, T const *lhs, T const *rhs, size_t length, float epsilon) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (; idx < length; idx += gridDim.x * blockDim.x) {
        if (!CompareFunc(lhs[idx], rhs[idx], epsilon)) {
            *equal = 0;
            return;
        }
    }
}

template <class T, bool REL>
bool compare_all_close(device_memory_t, const T *lhs, const T *rhs, size_t length, C<REL>, float epsilon) {
    constexpr auto compare_func = REL ? relative_close<T> : absolute_close<T>;

    int hres[1] = {1};
    auto dres = (int *)device_memory_t::malloc(sizeof(int));
    memory_copy<device_memory_t, host_memory_t>(dres, hres, sizeof(int));

    constexpr int nthreads = 1024;
    const int nblocks = (length + nthreads - 1) / nthreads;
    compare_all_close_kernel<T, compare_func><<<nblocks, nthreads>>>(dres, lhs, rhs, length, epsilon);
    sync_device();

    memory_copy<host_memory_t, device_memory_t>(hres, dres, sizeof(int));
    device_memory_t::free(dres);
    return *hres;
}

#define DEVICE_COMPARE_ALL_CLOSE(T)                                                                  \
    template bool compare_all_close(device_memory_t, const T *, const T *, size_t, C<false>, float); \
    template bool compare_all_close(device_memory_t, const T *, const T *, size_t, C<true>, float);

DEVICE_COMPARE_ALL_CLOSE(float);
DEVICE_COMPARE_ALL_CLOSE(double);
DEVICE_COMPARE_ALL_CLOSE(__half);
DEVICE_COMPARE_ALL_CLOSE(__hip_bfloat16);
} // namespace moke
