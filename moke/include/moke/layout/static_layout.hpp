#pragma once
#include "moke/common.hpp"
#include "moke/meta.hpp"

namespace moke {
template <class Shape, class Stride, size_t Size, int DimCons>
struct static_layout;

template <std::integral auto... Shapes, std::integral auto... Strides, size_t Size, int DimCons>
struct static_layout<constant_tuple<Shapes...>, constant_tuple<Strides...>, Size, DimCons> {
    using offset_t = size_t;

    using shape = constant_tuple<Shapes...>;
    using stride = constant_tuple<Strides...>;

    static_assert(sizeof...(Shapes) == sizeof...(Strides));
    constexpr static size_t rank = sizeof...(Shapes);
    constexpr static size_t size = Size;
    constexpr static bool empty = (size == 0);
    constexpr static int dim_cons = DimCons;

    MOKE_CONSTEXPR offset_t operator()(std::integral auto... indices) const noexcept {
        static_assert(sizeof...(indices) <= rank, "too many indices for this layout");
        return offset_of<0>(indices...);
    }

private:
    template <size_t IDX>
    MOKE_CONSTEXPR offset_t offset_of() const noexcept { return offset_t(0); }

    template <size_t IDX>
    MOKE_CONSTEXPR offset_t offset_of(std::integral auto index, std::integral auto... rest) const noexcept {
        return offset_t(index) * offset_t(get_value<stride, IDX>()) + offset_of<IDX + 1>(rest...);
    }
};

namespace meta {
    template <class Shape>
    struct right_major_layout;

    template <std::integral auto N>
    struct right_major_layout<constant_tuple<N>> {
        using shape = constant_tuple<N>;
        using stride = constant_tuple<1>;
    };

    template <std::integral auto N, std::integral auto... Rest>
    struct right_major_layout<constant_tuple<N, Rest...>> {
        using rest_shape = right_major_layout<constant_tuple<Rest...>>::shape;
        using rest_stride = right_major_layout<constant_tuple<Rest...>>::stride;
        using curr_stride = constant_tuple<get_value<rest_shape, 0>() * get_value<rest_stride, 0>()>;

        using shape = constant_tuple<N, Rest...>;
        using stride = concat_t<curr_stride, rest_stride>;
    };
} // namespace meta

template <std::integral auto... Shapes>
MOKE_CONSTEVAL auto make_right_major_layout(constant_tuple<Shapes...>) {
    static_assert(sizeof...(Shapes) > 0);
    using shape = constant_tuple<Shapes...>;
    using stride = meta::right_major_layout<shape>::stride;
    constexpr int rank = sizeof...(Shapes);
    constexpr size_t size = get_value<shape, 0>() * get_value<stride, 0>();
    return static_layout<shape, stride, size, rank - 1>{};
}

template <std::integral auto... Shapes>
MOKE_CONSTEVAL auto make_right_major_layout(constant<Shapes>...) {
    return make_right_major_layout(constant_tuple<Shapes...>{});
}

template <std::integral auto... Shapes>
MOKE_CONSTEVAL auto make_layout(constant_tuple<Shapes...> shape) {
    return make_right_major_layout(shape);
}

template <std::integral auto... Shapes>
MOKE_CONSTEVAL auto make_layout(constant<Shapes>... shapes) {
    return make_right_major_layout(shapes...);
}

namespace meta {
    template <std::integral auto Stride, std::integral auto... Shapes>
    struct left_major_layout;

    template <std::integral auto Stride>
    struct left_major_layout<Stride> {
        using stride = constant_tuple<>;
    };

    template <std::integral auto Stride, std::integral auto Shape, std::integral auto... Shapes>
    struct left_major_layout<Stride, Shape, Shapes...> {
        constexpr static auto curr_stride = Stride;
        constexpr static auto next_stride = Shape * Stride;
        using shape = constant_tuple<Shape, Shapes...>;
        using stride = concat_t<constant_tuple<curr_stride>, typename left_major_layout<next_stride, Shapes...>::stride>;
    };
} // namespace meta

template <std::integral auto... Shapes>
MOKE_CONSTEVAL auto make_left_major_layout(constant_tuple<Shapes...>) {
    using shape = constant_tuple<Shapes...>;
    using stride = meta::left_major_layout<1, Shapes...>::stride;
    constexpr int rank = sizeof...(Shapes);
    constexpr size_t size = get_value<shape, rank - 1>() * get_value<stride, rank - 1>();
    return static_layout<shape, stride, size, 0>{};
}

template <std::integral auto... Shapes>
MOKE_CONSTEVAL auto make_left_major_layout(constant<Shapes>...) {
    return make_left_major_layout(constant_tuple<Shapes...>{});
}
} // namespace moke
