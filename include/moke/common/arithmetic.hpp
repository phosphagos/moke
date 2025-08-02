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
} // namespace moke
