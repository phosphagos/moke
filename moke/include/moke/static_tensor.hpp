#pragma once
#include "moke/common.hpp"
#include "moke/layout/static_layout.hpp"

namespace moke {
template <class T, class Layout>
class static_tensor;

template <class T, std::integral auto... Shapes, std::integral auto... Strides>
class static_tensor<T, static_layout<constant_tuple<Shapes...>, constant_tuple<Strides...>>> {
public:
    using layout_t = static_layout<constant_tuple<Shapes...>, constant_tuple<Strides...>>;
    using shape_t = typename layout_t::shape;
    using stride_t = typename layout_t::stride;
    using value_t = T;

    constexpr static int RANK = layout_t::rank;

private:
    T *m_data{nullptr};

public:
    template <class, class> friend class static_tensor;

    MOKE_CONSTEVAL static_tensor() noexcept = default;

    MOKE_CONSTEXPR static_tensor(T *data) noexcept : m_data{data} {}

    template <class U> requires(std::is_convertible_v<U *, T *>)
    MOKE_INLINE static_tensor(const static_tensor<U, layout_t> &other) noexcept
            : m_data{other.m_data} {}

    MOKE_CONSTEVAL static auto rank() noexcept { return RANK; }

    MOKE_CONSTEVAL static auto size() noexcept { return layout_t::size; }

    MOKE_CONSTEVAL static auto bytes() noexcept { return size() * sizeof(T); }

    MOKE_CONSTEVAL static auto layout() noexcept { return layout_t{}; }

    template <std::integral auto Dim>
    MOKE_CONSTEVAL static auto shape(C<Dim> = {}) noexcept { return get_value<shape_t, Dim>(); }

    template <std::integral auto Dim>
    MOKE_CONSTEVAL static auto stride(C<Dim> = {}) noexcept { return get_value<stride_t, Dim>(); }

    MOKE_CONSTEXPR bool empty() const noexcept { return m_data == nullptr || layout_t::empty; }

    MOKE_CONSTEXPR T *data() const noexcept { return m_data; }
    // partial indexing yields a pointer to the start of the addressed sub-tensor
    template <std::integral... Coord> requires(sizeof...(Coord) < RANK)
    MOKE_CONSTEXPR T *operator()(const Coord &...coord) const noexcept { return m_data + layout_t{}(coord...); }

    // full indexing yields a reference to the element
    template <std::integral... Coord> requires(sizeof...(Coord) == RANK)
    MOKE_CONSTEXPR T &operator()(const Coord &...coord) const noexcept { return m_data[layout_t{}(coord...)]; }
};

template <class T, std::integral auto... Shapes>
MOKE_CONSTEXPR auto make_left_major_tensor(T *data, constant_tuple<Shapes...>) {
    using layout_t = decltype(make_left_major_layout(constant_tuple<Shapes...>{}));
    return static_tensor<T, layout_t>{data};
}

template <class T, std::integral auto... Shapes>
MOKE_CONSTEXPR auto make_left_major_tensor(T *data, constant<Shapes>...) {
    return make_left_major_tensor(data, constant_tuple<Shapes...>{});
}

template <class T, std::integral auto... Shapes>
MOKE_CONSTEXPR auto make_right_major_tensor(T *data, constant_tuple<Shapes...>) {
    using layout_t = decltype(make_right_major_layout(constant_tuple<Shapes...>{}));
    return static_tensor<T, layout_t>{data};
}

template <class T, std::integral auto... Shapes>
MOKE_CONSTEXPR auto make_right_major_tensor(T *data, constant<Shapes>...) {
    return make_right_major_tensor(data, constant_tuple<Shapes...>{});
}

template <class T, std::integral auto... Shapes>
MOKE_CONSTEXPR auto make_tensor(T *data, constant_tuple<Shapes...> shape) {
    return make_right_major_tensor(data, shape);
}

template <class T, std::integral auto... Shapes>
MOKE_CONSTEXPR auto make_tensor(T *data, constant<Shapes>... shapes) {
    return make_right_major_tensor(data, shapes...);
}
} // namespace moke
