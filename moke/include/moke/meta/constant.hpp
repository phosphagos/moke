#pragma once
#include "moke/common.hpp"
#include <type_traits>
#include <climits>

namespace moke {
// compile-time constant type
template <auto N>
struct C : std::integral_constant<std::remove_cvref_t<decltype(N)>, N> {};

// alias constant<N> to C<N>
template <auto N>
using constant = C<N>;

// compile-time constant type for bool
using true_t = C<true>;
using false_t = C<false>;

// compile-time constant value for bool
constexpr auto TRUE = C<true>{};
constexpr auto FALSE = C<false>{};

namespace meta {
    template <size_t value, char digit>
    MOKE_CONSTEVAL std::integral auto parse_integer_digit() noexcept {
        static_assert(digit >= '0' && digit <= '9' || digit == '\'');
        if constexpr (digit == '\'') {
            return value;
        } else {
            return (value * 10) + (digit - '0');
        }
    }

    template <size_t value>
    MOKE_CONSTEVAL auto make_integral_constant() noexcept {
        static_assert(value <= LONG_LONG_MAX);
        if constexpr (value <= INT_MAX) {
            return C<int(value)>{};
        } else if constexpr(value <= LONG_MAX) {
            return C<long(value)>{};
        } else {
            return C<(long long)(value)>{};
        }
    }

    template <size_t value, char digit, char... digits>
    MOKE_CONSTEVAL auto parse_integral_constant() noexcept {
        constexpr size_t next_value = parse_integer_digit<value, digit>();
        if constexpr (sizeof...(digits) == 0) {
            return make_integral_constant<next_value>();
        } else {
            return parse_integral_constant<next_value, digits...>();
        }
    }
} // namespace meta
} // namespace moke

namespace moke::literals {
template <char... digits>
MOKE_CONSTEVAL auto operator""_ic() noexcept { return meta::parse_integral_constant<0, digits...>(); }
} // namespace moke::literals
