#include "moke/common.hpp"
#include "moke/runtime.hpp"
#include "moke/native/memory.hpp"

namespace moke {
MOKE_KERNEL void memcmp_kernel(int *equal, const char *lhs, const char *rhs, size_t length) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (; idx < length; idx += gridDim.x * blockDim.x) {
        if (lhs[idx] != rhs[idx]) { return (*equal = 0, void()); }
    }
}

bool device_memory_t::memcmp(const void *lhs, const void *rhs, size_t length) {
    int hres[1] = {1};
    auto dres = (int *)device_memory_t::malloc(sizeof(int));
    memory_copy<device_memory_t, host_memory_t>(dres, hres, sizeof(int));

    constexpr int nthreads = 1024;
    const int nblocks = (length + nthreads - 1) / nthreads;
    memcmp_kernel<<<nblocks, nthreads>>>(dres, (const char *)lhs, (const char *)rhs, length);
    sync_device();

    memory_copy<host_memory_t, device_memory_t>(hres, dres, sizeof(int));
    device_memory_t::free(dres);
    return *hres;
}

template <class T>
MOKE_KERNEL void memset_kernel(T *dest, T value, size_t n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) { dest[idx] = value; }
}

void device_memory_t::memset(void *dest, uint64_t value, size_t n) {
    constexpr int nthreads = 1024;
    const int nblocks = (n + nthreads - 1) / nthreads;
    memset_kernel<<<nblocks, nthreads>>>((uint64_t *)(dest), value, n);
}
} // namespace moke
