#pragma once
#include "moke/meta/constant.hpp"

namespace moke {
template <class... Ts> struct type_tuple {
    constexpr static int size = sizeof...(Ts);
};

template <auto... Ns>
using constant_tuple = type_tuple<constant<Ns>...>;

namespace meta {
    template <class T, int Idx> struct get;

    template <template <class...> class Tuple, class T, class... Ts, int Idx>
    struct get<Tuple<T, Ts...>, Idx> {
        using type = typename get<Tuple<Ts...>, Idx - 1>::type;
    };

    template <template <class...> class Tuple, class T, class... Ts>
    struct get<Tuple<T, Ts...>, 0> {
        using type = T;
    };

    template <template <class...> class Tuple, int Idx>
    struct get<Tuple<>, Idx> {
        static_assert(Idx >= 0, "Idx out of range");
    };
} // namespace meta

template <class T, int Idx>
using get_t = typename meta::get<T, Idx>::type;

template <class T, int Idx>
constexpr auto get_v = get_t<T, Idx>::value;

template <class Tuple, int Idx>
consteval auto get_value(Tuple = {}, C<Idx> = {}) { return get_t<Tuple, Idx>::value; }

namespace meta {
    template <class U, class V>
    struct concat;

    template <template <class...> class Tuple, class... Us, class... Vs>
    struct concat<Tuple<Us...>, Tuple<Vs...>> {
        using type = Tuple<Us..., Vs...>;
    };

    template <template <class...> class Tuple, class... Us, class V>
    struct concat<Tuple<Us...>, V> {
        using type = Tuple<Us..., V>;
    };
} // namespace meta

template <class U, class V>
using concat_t = typename meta::concat<U, V>::type;
} // namespace moke
