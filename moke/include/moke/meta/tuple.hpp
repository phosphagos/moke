#pragma once
#include "moke/meta/constant.hpp"
#include <type_traits>
#include <utility>

namespace moke {
template <class... Ts>
struct type_tuple {
    constexpr static size_t size = sizeof...(Ts);
};

template <auto... Ns>
struct constant_tuple {
    constexpr static size_t size = sizeof...(Ns);
};

//
// meta implementation of get_t
//
namespace meta {
    template <class T, index_t Idx>
    struct get;

    template <class T, class... Ts, index_t Idx>
    struct get<type_tuple<T, Ts...>, Idx> {
        using type = typename get<type_tuple<Ts...>, Idx - 1>::type;
    };

    template <class T, class... Ts>
    struct get<type_tuple<T, Ts...>, 0> {
        using type = T;
    };

    template <index_t Idx>
    struct get<type_tuple<>, Idx> {
        static_assert(Idx >= 0, "Idx out of range");
    };
} // namespace meta

template <class T, index_t Idx>
using get_t = typename meta::get<T, Idx>::type;

//
// meta implementation of get_value
//
namespace meta {
    template <auto T, auto... Ts, index_t Idx>
    struct get<constant_tuple<T, Ts...>, Idx> {
        using type = typename get<constant_tuple<Ts...>, Idx - 1>::type;
        constexpr static auto value = type::value;
    };

    template <auto T, auto... Ts>
    struct get<constant_tuple<T, Ts...>, 0> {
        using type = constant<T>;
        constexpr static auto value = constant<T>{};
    };

    template <index_t Idx>
    struct get<constant_tuple<>, Idx> {
        static_assert(Idx >= 0, "Idx out of range");
    };
} // namespace meta

template <class CTuple, std::integral auto Idx>
consteval auto get_value(CTuple = {}, C<Idx> = {}) { return get_t<CTuple, Idx>::value; }

//
// meta implementation of concat_t
//
namespace meta {
    template <class U, class V>
    struct concat;

    template <auto... Us, auto... Vs>
    struct concat<constant_tuple<Us...>, constant_tuple<Vs...>> {
        using type = constant_tuple<Us..., Vs...>;
    };

    template <auto... Us, auto V>
    struct concat<constant_tuple<Us...>, constant<V>> {
        using type = constant_tuple<Us..., V>;
    };
} // namespace meta

template <class U, class V>
using concat_t = typename meta::concat<U, V>::type;

//
// meta implementation of shift_t
//
namespace meta {
    template <class T>
    struct shift;

    template <auto U, auto... Vs>
    struct shift<constant_tuple<U, Vs...>> {
        using type = constant_tuple<Vs...>;
    };
} // namespace meta

template <class T>
using shift_t = typename meta::shift<T>::type;

//
// meta implementation of swap_t
//
namespace meta {
    template <index_t M, index_t N, index_t Idx>
    struct swap_index {
        constexpr static index_t value = Idx;
    };

    template <index_t M, index_t N>
    struct swap_index<M, N, M> {
        constexpr static index_t value = N;
    };

    template <index_t M, index_t N>
    struct swap_index<M, N, N> {
        constexpr static index_t value = M;
    };

    template <index_t M>
    struct swap_index<M, M, M> {
        constexpr static index_t value = M;
    };

    template <class Tuple, class Seq, index_t M, index_t N>
    struct swap_impl;

    template <class T, index_t... Is, index_t M, index_t N>
    struct swap_impl<T, std::integer_sequence<index_t, Is...>, M, N> {
        using type = constant_tuple<get_t<T, swap_index<M, N, Is>::value>::value...>;
    };

    template <class T, index_t M, index_t N>
    struct swap;

    template <auto... Ns, index_t M, index_t N>
    struct swap<constant_tuple<Ns...>, M, N> {
        using tuple = constant_tuple<Ns...>;
        using sequence = std::make_integer_sequence<index_t, sizeof...(Ns)>;
        using type = typename swap_impl<tuple, sequence, M, N>::type;
    };
} // namespace meta

template <class T, index_t M, index_t N>
using swap_t = typename meta::swap<T, M, N>::type;
} // namespace moke
