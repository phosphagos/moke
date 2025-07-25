#include "moke/common/memory.hpp"
#include "moke/common/runtime.hpp"
#include <hip/hip_runtime.h>

namespace moke {
void DeviceAllocator::allocate(void **ptr, size_t size) {
    CheckStatus(hipMalloc(ptr, size));
}

void DeviceAllocator::deallocate(void *ptr) {
    if (ptr == nullptr) { return; }
    CheckStatus(hipFree(ptr));
}

void HostAllocator::allocate(void **ptr, size_t size) {
    CheckStatus(hipHostAlloc(ptr, size));
}

void HostAllocator::deallocate(void *ptr) {
    if (ptr == nullptr) { return; }
    CheckStatus(hipHostFree(ptr));
}

void Memcpy(void *dest, const void *source, size_t size) {
    CheckStatus(hipMemcpy(dest, source, size, hipMemcpyDefault));
}
} // namespace moke
