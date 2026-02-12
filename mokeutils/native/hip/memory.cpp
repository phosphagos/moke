#include "moke/native/memory.hpp"
#include "moke/runtime.hpp"

#include <hip/hip_runtime.h>

namespace moke {
void *device_memory_t::malloc(size_t size) {
    void *pointer{nullptr};
    check_status(hipMalloc(&pointer, size));
    return pointer;
}

void device_memory_t::free(void *pointer) {
    check_status(hipFree(pointer));
}

void device_memory_t::memset(void *dest, uint8_t value, size_t n) {
    check_status(hipMemset(dest, value, n));
}

void device_memory_t::memset(void *dest, uint16_t value, size_t n) {
    check_status(hipMemsetD16(dest, value, n));
}

void device_memory_t::memset(void *dest, uint32_t value, size_t n) {
    check_status(hipMemsetD32(dest, value, n));
}

template <> void memory_copy<device_memory_t>(void *dest, const void *src, size_t size) {
    check_status(hipMemcpy(dest, src, size, hipMemcpyDeviceToDevice));
}

template <> void memory_copy<device_memory_t, host_memory_t>(void *dest, const void *src, size_t size) {
    check_status(hipMemcpy(dest, src, size, hipMemcpyHostToDevice));
}

template <> void memory_copy<host_memory_t, device_memory_t>(void *dest, const void *src, size_t size) {
    check_status(hipMemcpy(dest, src, size, hipMemcpyDeviceToHost));
}
} // namespace moke
