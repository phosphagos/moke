#pragma once
#include <type_traits>

namespace moke {
// compile-time constant type
template <auto N>
struct C : std::integral_constant<std::remove_cvref_t<decltype(N)>, N> {};

// compile-time constant value
template <auto N> constexpr C<N> K{};

// alias constant<N> to C<N>
template <auto N>
using constant = C<N>;

// compile-time constant type for bool
using true_t = C<true>;
using false_t = C<false>;

// compile-time constant value for bool
constexpr auto TRUE = C<true>{};
constexpr auto FALSE = C<false>{};
} // namespace moke
