#pragma once
#include <type_traits>

namespace moke {
// compile-time constant
template <auto N>
struct C : std::integral_constant<std::remove_cvref_t<decltype(N)>, N> {};

template <auto N> using constant = C<N>;

// compile-time constant type for bool
using true_t = C<true>;
using false_t = C<false>;

// compile-time constant value for bool
constexpr auto TRUE = C<true>{};
constexpr auto FALSE = C<false>{};
} // namespace moke
