#include "moke/common.hpp"
#include "moke/native.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

namespace moke {
void *device_memory_t::malloc(size_t size) {
    void *pointer{nullptr};
    check_status(cudaMalloc(&pointer, size));
    return pointer;
}

void device_memory_t::free(void *pointer) {
    check_status(cudaFree(pointer));
}

void device_memory_t::memset(void *dest, uint8_t value, size_t n) {
    check_status(cudaMemset(dest, value, n));
}

void device_memory_t::memset(void *dest, uint16_t value, size_t n) {
    check_status(cuMemsetD16((CUdeviceptr)dest, value, n));
}

void device_memory_t::memset(void *dest, uint32_t value, size_t n) {
    check_status(cuMemsetD32((CUdeviceptr)dest, value, n));
}

template <> void memory_copy<device_memory_t>(void *dest, const void *src, size_t size) {
    check_status(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice));
}

template <> void memory_copy<device_memory_t, host_memory_t>(void *dest, const void *src, size_t size) {
    check_status(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
}

template <> void memory_copy<host_memory_t, device_memory_t>(void *dest, const void *src, size_t size) {
    check_status(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
}
} // namespace moke
