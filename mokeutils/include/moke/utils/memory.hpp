#pragma once
#include "moke/common.hpp"
#include "moke/native/memory.hpp"
#include <bit>

namespace moke {
template <class T, class memory_space_t>
T *memory_alloc(memory_space_t, size_t length) {
    return static_cast<T *>(memory_space_t::malloc(length * sizeof(T)));
}

template <class memory_space_t>
void memory_free(memory_space_t, void *pointer) { return memory_space_t::free(pointer); }

template <class memory_space_t, class T>
void memory_copy(memory_space_t, T *dest, const T *src, size_t length) {
    static_assert(std::is_trivially_copyable_v<T>);
    return memory_copy<memory_space_t>(dest, src, length * sizeof(T));
}

template <class dest_memory_t, class src_memory_t, class T>
void memory_copy(dest_memory_t, src_memory_t, T *dest, const T *src, size_t length) {
    static_assert(std::is_trivially_copyable_v<T>);
    return memory_copy<dest_memory_t, src_memory_t>(dest, src, length * sizeof(T));
}

template <class memory_space_t, class T>
void memory_set(memory_space_t, T *dest, T value, size_t size) {
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
    if constexpr (sizeof(T) == 1) {
        memory_space_t::memset(dest, std::bit_cast<uint8_t>(value), size);
    } else if constexpr (sizeof(T) == 2) {
        memory_space_t::memset(dest, std::bit_cast<uint16_t>(value), size);
    } else if constexpr (sizeof(T) == 4) {
        memory_space_t::memset(dest, std::bit_cast<uint32_t>(value), size);
    } else if constexpr (sizeof(T) == 8) {
        memory_space_t::memset(dest, std::bit_cast<uint64_t>(value), size);
    }
}

template <class memory_space_t, class T, class V>
void memory_set(memory_space_t mem_space, T *dest, V value, size_t size) {
    static_assert(std::is_trivially_copyable_v<T>);
    return memory_set(mem_space, dest, T(value), size);
}

template <class T, class memory_space_t>
bool memory_compare(memory_space_t, const T *lhs, const T *rhs, size_t length) {
    return memory_space_t::memcmp(lhs, rhs, length * sizeof(T));
}

template <class T>
T *device_malloc(size_t length) { return memory_alloc<T>(device_memory, length); }

template <class T>
T *host_malloc(size_t length) { return memory_alloc<T>(host_memory, length); }

template <class = void>
void device_free(void *pointer) { return memory_free(device_memory, pointer); }

template <class = void>
void host_free(void *pointer) { return memory_free(host_memory, pointer); }
} // namespace moke
