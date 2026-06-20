#pragma once
#include "moke/meta.hpp"

namespace moke {
namespace meta {

    template <class T, class List> struct combine;

    template <template <class...> class TypePack, class T, class... Ts>
    struct combine<T, TypePack<Ts...>> {
        using type = TypePack<TypePack<T, Ts>...>;
    };

    template <template <class...> class TypePack, class... Es, class... Ts>
    struct combine<TypePack<Es...>, TypePack<Ts...>> {
        using type = TypePack<TypePack<Es..., Ts>...>;
    };

    template <class Left, class Right>
    struct product_binary;

    template <class... Lists>
    struct product;

    template <class T, class... Ts, class Right> struct product_binary<type_tuple<T, Ts...>, Right> {
        using left_part = typename combine<T, Right>::type;
        using right_part = typename product_binary<type_tuple<Ts...>, Right>::type;
        using type = typename concat<left_part, right_part>::type;
    };

    template <class Right> struct product_binary<type_tuple<>, Right> {
        using type = type_tuple<>;
    };

    template <class Left, class Right> struct product<Left, Right> {
        using type = typename product_binary<Left, Right>::type;
    };

    template <class Left, class Right, class... Rests>
    struct product<Left, Right, Rests...> {
        using type = typename product<typename product<Left, Right>::type, Rests...>::type;
    };

    template <template <class...> class U, class V> struct convert;

    template <template <class...> class U, template <class...> class V, class... Ts>
    struct convert<U, V<Ts...>> {
        using type = U<Ts...>;
    };
} // namespace meta

template <class... Lists>
using product_t = typename meta::product<Lists...>::type;

template <template <class...> class U, class V>
using convert_t = typename meta::convert<U, V>::type;
} // namespace moke
