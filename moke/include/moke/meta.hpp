#pragma once
#include <type_traits>

namespace moke {
// compile-time constant
template <auto N>
struct C : std::integral_constant<std::remove_cvref_t<decltype(N)>, N> {};

// compile-time constant type and value for bool (true/false)
using true_t = C<true>;
using false_t = C<false>;
constexpr auto TRUE = C<true>{};
constexpr auto FALSE = C<false>{};

template <auto N> using constant = C<N>;

template <class... Ts> struct type_tuple {
    constexpr static int size = sizeof...(Ts);
};

template <auto... Ns>
using constant_tuple = type_tuple<constant<Ns>...>;

namespace meta {
    template <class T, int Idx> struct get;

    template <template <class...> class TypePack, class T, class... Ts, int Idx>
    struct get<TypePack<T, Ts...>, Idx> {
        using type = typename get<TypePack<Ts...>, Idx - 1>::type;
    };

    template <template <class...> class TypePack, class T, class... Ts>
    struct get<TypePack<T, Ts...>, 0> {
        using type = T;
    };

    template <template <class...> class TypePack, int Idx>
    struct get<TypePack<>, Idx> {
        static_assert(Idx >= 0, "Idx out of range");
    };
} // namespace meta

template <class T, int Idx>
using get = meta::get<T, Idx>;

template <class T, int Idx>
using get_type = typename meta::get<T, Idx>::type;

template <class T, int Idx>
constexpr auto get_value = meta::get<T, Idx>::type::value;

namespace meta {
    template <class T1, class T2>
    struct concat;

    template <template <class...> class TypePack, class... T1, class... Ts>
    struct concat<TypePack<T1...>, TypePack<Ts...>> {
        using type = TypePack<T1..., Ts...>;
    };

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

    template <template <class ...> class U, class V> struct convert;

    template <template <class ...> class U, template <class ...> class V, class ...Ts>
    struct convert<U, V<Ts...>> { using type = U<Ts...>; };
} // namespace meta

template <class Left, class Right>
using concat_t = typename meta::concat<Left, Right>::type;

template <class... Lists>
using product_t = typename meta::product<Lists...>::type;

template <template <class ...> class U, class V>
using convert_t = typename meta::convert<U, V>::type;
} // namespace moke
