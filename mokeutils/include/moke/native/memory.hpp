#pragma once
#include "moke/common.hpp"
#include "moke/dtype.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace moke {
constexpr struct host_memory_t {
    static void *malloc(size_t size) { return std::malloc(size); }
    static void free(void *pointer) { std::free(pointer); }
    static void memset(void *dest, uint8_t value, size_t n) { std::fill_n((uint8_t *)dest, n, value); }
    static void memset(void *dest, uint16_t value, size_t n) { std::fill_n((uint16_t *)dest, n, value); }
    static void memset(void *dest, uint32_t value, size_t n) { std::fill_n((uint32_t *)dest, n, value); }
    static void memset(void *dest, uint64_t value, size_t n) { std::fill_n((uint64_t *)dest, n, value); }
    static bool memcmp(const void *lhs, const void *rhs, size_t n) { return 0 == std::memcmp(lhs, rhs, n); }
} host_memory;

constexpr struct device_memory_t {
    static void *malloc(size_t size);
    static void free(void *pointer);
    static void memset(void *dest, uint8_t value, size_t n);
    static void memset(void *dest, uint16_t value, size_t n);
    static void memset(void *dest, uint32_t value, size_t n);
    static void memset(void *dest, uint64_t value, size_t n);
    static bool memcmp(const void *lhs, const void *rhs, size_t n);
} device_memory;

template <class dest_memory_t, class src_memory_t = dest_memory_t>
void memory_copy(void *dest, const void *src, size_t size);

template <> inline void memory_copy<host_memory_t>(void *dest, const void *src, size_t size) {
    std::memcpy(dest, src, size);
}
} // namespace moke
