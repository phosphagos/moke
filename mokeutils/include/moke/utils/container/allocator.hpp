#pragma once
#include "moke/utils/memory.hpp"

namespace moke {
template <class T, class Memory>
struct allocator {
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    using memory_space = Memory;

    constexpr static memory_space mem_space() { return memory_space{}; }
    pointer allocate(size_t length) { return memory_alloc<T>(mem_space(), length); }
    void deallocate(pointer pbuf) { return memory_free(mem_space(), pbuf); }
    void deallocate(pointer pbuf, size_t length) { return memory_free(mem_space(), pbuf); }
};
} // namespace moke
