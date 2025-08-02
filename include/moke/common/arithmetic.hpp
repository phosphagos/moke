#pragma once
#include "moke/common/def.hpp"
#include "moke/common/runtime.hpp"
#include <concepts>

namespace moke {
/// @brief check whether `n` is pow of 2
MOKE_TRIVIAL bool is_pow2(std::integral auto n) {
    return (n & (n - 1)) == 0;
}

/// @brief compute ceil(m / n)
MOKE_TRIVIAL auto ceil_div(std::integral auto m, std::integral auto n) {
    return (m + n - 1) / n;
}

/// @brief pad down size to alignment
MOKE_TRIVIAL auto pad_down(std::integral auto size, std::integral auto align) {
    return size / align * align;
}

/// @brief pad up size to alignment
MOKE_TRIVIAL auto pad_up(std::integral auto size, std::integral auto align) {
    return (size + align - 1) / align * align;
}

/// @brief pad down size to alignment, which is a constant value and pow of 2
template <auto ALIGN> requires(is_pow2(ALIGN))
MOKE_TRIVIAL auto pad_down(std::integral auto size) {
    return size & ~(ALIGN - 1);
}

/// @brief pad up size to alignment, which is a constant value and pow of 2
template <auto ALIGN> requires(is_pow2(ALIGN))
MOKE_TRIVIAL auto pad_up(std::integral auto size) {
    return (size + ALIGN - 1) & ~(ALIGN - 1);
}

template <class T, size_t N>
consteval size_t length_of(const T (&arr)[N]) { return N; }

#if defined(MOKE_PLATFORM_CUDA) || defined(MOKE_PLATFORM_HIP)
MOKE_TRIVIAL dim3 operator+(dim3 m, dim3 n) {
    return {m.x + n.x, m.y + n.y, m.z + n.z};
}

MOKE_TRIVIAL dim3 operator*(dim3 m, dim3 n) {
    return {m.x * n.x, m.y * n.y, m.z * n.z};
}

MOKE_TRIVIAL dim3 operator/(dim3 m, dim3 n) {
    return {m.x / n.x, m.y / n.y, m.z / n.z};
}

MOKE_TRIVIAL dim3 ceil_div(dim3 m, dim3 n) {
    return {ceil_div(m.x, n.x), ceil_div(m.y, n.y), ceil_div(m.z, n.z)};
}
#endif // MOKE_PLATFORM_CUDA or MOKE_PLATFORM_HIP
} // namespace moke
