#pragma once
#include "moke/meta/constant.hpp"
#include "moke/meta/type_tuple.hpp"
#include <concepts>
#include <type_traits>

namespace moke {
template <auto... Ns>
using constant_tuple = type_tuple<constant<Ns>...>;

namespace meta {
    template <std::integral Ret, std::size_t I, class Tuple, std::integral Index>
    Ret get_value_impl(Index index) {
        if constexpr (I == Tuple::size) {
            return Ret(0);
        } else if (index == I) {
            return static_cast<Ret>(moke::get_t<Tuple, I>::value);
        } else {
            return get_value_impl<Ret, I + 1, Tuple>(index);
        }
    }
} // namespace meta

template <class Tuple, int Idx>
consteval auto get_value(Tuple = {}, C<Idx> = {}) { return get_t<Tuple, Idx>::value; }

template <std::integral Ret = int, class Tuple, std::integral Index>
Ret get_value(Tuple tuple, Index index) {
    if (index < 0) { return Ret(0); }
    return meta::get_value_impl<Ret, 0, Tuple>(index);
}
} // namespace moke
