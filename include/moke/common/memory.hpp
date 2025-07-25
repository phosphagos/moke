#pragma once
#include "moke/common/def.hpp"

namespace moke {
struct DeviceAllocator {
    static void allocate(void **ptr, size_t size);
    static void deallocate(void *ptr);
};

struct HostAllocator {
    static void allocate(void **ptr, size_t size);
    static void deallocate(void *ptr);
};

void Memcpy(void *dest, const void *source, size_t size);
} // namespace moke
