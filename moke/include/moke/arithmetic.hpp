#pragma once
#include "moke/common.hpp"
#include "moke/dtype.hpp"
#include "moke/meta.hpp"
#include "moke/meta/constant.hpp"
#include <climits>
#include <utility>

//
// arithmetic utilities
//

namespace moke {
/// @brief check whether `n` is pow of 2
MOKE_CONSTEXPR bool is_pow2(std::integral auto n) { return (n & (n - 1)) == 0; }

/// @brief compute ceil(m / n)
MOKE_CONSTEXPR auto ceil_div(std::integral auto m, std::integral auto n) { return (m + n - 1) / n; }

/// @brief compute minimum of m, n
MOKE_CONSTEXPR auto min(auto m, auto n) { return m < n ? m : n; }

/// @brief compute maximum of m, n
MOKE_CONSTEXPR auto max(auto m, auto n) { return m > n ? m : n; }

/// @brief compute min(m, n) at compile time
template <std::integral auto M, std::integral auto N>
MOKE_CONSTEVAL C<min(M, N)> min(C<M> m, C<N> n) { return {}; }

/// @brief compute max(m, n) at compile time
template <std::integral auto M, std::integral auto N>
MOKE_CONSTEVAL C<max(M, N)> max(C<M> m, C<N> n) { return {}; }

/// @brief pad down size to alignment
template <std::integral auto ALIGN>
MOKE_CONSTEXPR auto pad_down(std::integral auto size, constant<ALIGN> align) {
    if constexpr (is_pow2(ALIGN)) {
        return size & ~(ALIGN - 1);
    } else {
        return size / ALIGN * ALIGN;
    }
}

/// @brief pad up size to alignment
template <std::integral auto ALIGN> requires(is_pow2(ALIGN))
MOKE_CONSTEXPR auto pad_up(std::integral auto size, constant<ALIGN> align) {
    if constexpr (is_pow2(align.value)) {
        return (size + ALIGN - 1) & ~(ALIGN - 1);
    } else {
        return (size + ALIGN - 1) / ALIGN * ALIGN;
    }
}
} // namespace moke

//
// utilities of lengths and sizes
//

namespace moke {
template <class T, size_t N>
using array_ref = const T (&)[N];

template <class T, size_t N>
MOKE_CONSTEVAL size_t length_of(array_ref<T, N>) { return N; }

template <class T>
MOKE_CONSTEVAL size_t bytes_of(T = std::declval<T>()) { return sizeof(T); }

template <class T>
MOKE_CONSTEVAL size_t bits_of(T = std::declval<T>()) { return sizeof(T) * CHAR_BIT; }
} // namespace moke

//
// utilities of dim3
//

#if defined(MOKE_PLATFORM_CUDA) || defined(MOKE_PLATFORM_HIP)
#include "moke/runtime.hpp"

namespace moke {
/// @brief create 1d coord of dim3(x)
MOKE_CONSTEXPR dim3 make_dim3(auto x) { return dim3{uint(x)}; }

/// @brief create 2d coord of dim3(y, x)
MOKE_CONSTEXPR dim3 make_dim3(auto y, auto x) { return dim3{uint(x), uint(y)}; }

/// @brief create 3d coord of dim3(z, y, x)
MOKE_CONSTEXPR dim3 make_dim3(auto z, auto y, auto x) { return dim3{uint(x), uint(y), uint(z)}; }

/// @brief compute ceil(m / n)
MOKE_CONSTEXPR dim3 ceil_div(dim3 m, dim3 n) { return {ceil_div(m.x, n.x), ceil_div(m.y, n.y), ceil_div(m.z, n.z)}; }
} // namespace moke
#endif // MOKE_PLATFORM_CUDA or MOKE_PLATFORM_HIP
