#pragma once
#include "moke/meta.hpp"

namespace moke {
//
// meta implementation of concat_t for type_tuple
//
namespace meta {
    template <class... Us, class... Vs>
    struct concat<type_tuple<Us...>, type_tuple<Vs...>> {
        using type = type_tuple<Us..., Vs...>;
    };

    template <class... Us, class V>
    struct concat<type_tuple<Us...>, V> {
        using type = type_tuple<Us..., V>;
    };
} // namespace meta

//
// meta implementation of product_t
//
namespace meta {
    template <class T, class Tuple> struct connect;

    template <class T, class... Ts>
    struct connect<T, type_tuple<Ts...>> {
        using type = type_tuple<type_tuple<T, Ts>...>;
    };

    template <class... Us, class... Vs>
    struct connect<type_tuple<Us...>, type_tuple<Vs...>> {
        using type = type_tuple<type_tuple<Us..., Vs>...>;
    };

    template <class U, class V>
    struct product_binary;

    template <class... Ts>
    struct product;

    template <class T, class... Us, class V>
    struct product_binary<type_tuple<T, Us...>, V> {
        using left_part = typename connect<T, V>::type;
        using right_part = typename product_binary<type_tuple<Us...>, V>::type;
        using type = typename concat<left_part, right_part>::type;
    };

    template <class T>
    struct product_binary<type_tuple<>, T> {
        using type = type_tuple<>;
    };

    template <class U, class V>
    struct product<U, V> {
        using type = typename product_binary<U, V>::type;
    };

    template <class U, class V, class... Rs>
    struct product<U, V, Rs...> {
        using type = typename product<typename product<U, V>::type, Rs...>::type;
    };
} // namespace meta

template <class... Lists>
using product_t = typename meta::product<Lists...>::type;

//
// meta implementation of convert_t
//
namespace meta {
    template <template <class...> class U, class V>
    struct convert;

    template <template <class...> class U, template <class...> class V, class... Ts>
    struct convert<U, V<Ts...>> {
        using type = U<Ts...>;
    };
} // namespace meta

template <template <class...> class U, class V>
using convert_t = typename meta::convert<U, V>::type;
} // namespace moke
