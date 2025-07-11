#include "moke/common/memory.hpp"
#include "moke/common/runtime.hpp"
#include <cuda_runtime.h>

namespace moke {
void DeviceAllocator::allocate(void **ptr, size_t size) {
    CheckStatus(cudaMalloc(ptr, size));
}

void DeviceAllocator::deallocate(void *ptr) {
    if (ptr == nullptr) { return; }
    CheckStatus(cudaFree(ptr));
}

void HostAllocator::allocate(void **ptr, size_t size) {
    CheckStatus(cudaMallocHost(ptr, size));
}

void HostAllocator::deallocate(void *ptr) {
    if (ptr == nullptr) { return; }
    CheckStatus(cudaFreeHost(ptr));
}

void Memcpy(void *dest, const void *source, size_t size) {
    CheckStatus(cudaMemcpy(dest, source, size, cudaMemcpyDefault));
}
} // namespace moke
