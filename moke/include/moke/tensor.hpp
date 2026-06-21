#pragma once
#include "moke/common.hpp"
#include "moke/layout/left_major.hpp"
#include "moke/layout/right_major.hpp"

namespace moke {
template <class T, class Layout>
class basic_tensor {
public:
    constexpr static int RANK = Layout::RANK;
    using layout_t = Layout;

private:
    T *m_data;
    Layout m_layout;

public:
    template <class, class> friend class basic_tensor;

    MOKE_INLINE basic_tensor() noexcept : m_data{nullptr}, m_layout{} {}

    MOKE_INLINE basic_tensor(T *data, const std::integral auto &...shape) noexcept
            : m_data{data}, m_layout{shape...} {}

    MOKE_INLINE basic_tensor(T *data, Layout layout) noexcept
            : m_data{data}, m_layout{std::move(layout)} {}

    template <class U> requires(std::is_convertible_v<U *, T *>)
    MOKE_INLINE basic_tensor(const basic_tensor<U, Layout> &other) noexcept
            : m_data{other.m_data}, m_layout{other.m_layout} {}

    MOKE_INLINE auto rank() const noexcept { return RANK; }

    MOKE_INLINE auto size() const noexcept { return m_layout.size(); }

    MOKE_INLINE auto bytes() const noexcept { return size() * sizeof(T); }

    MOKE_INLINE auto shape(int dim) const noexcept { return m_layout.m_shape[dim]; }

    MOKE_INLINE auto stride(int dim) const noexcept { return m_layout.m_stride[dim]; }

    MOKE_INLINE bool empty() const noexcept { return m_data == nullptr || m_layout.empty(); }

    MOKE_INLINE T *data() const noexcept { return m_data; }

    MOKE_INLINE const Layout &layout() const noexcept { return m_layout; }

    template <std::integral... Coord> requires(sizeof...(Coord) < RANK)
    MOKE_INLINE T *operator()(const Coord &...coord) const noexcept { return m_data + m_layout(coord...); }

    template <std::integral... Coord> requires(sizeof...(Coord) == RANK)
    MOKE_INLINE T &operator()(const Coord &...coord) const noexcept { return m_data[m_layout(coord...)]; }
};

///
/// type alias and deducing hint for tensor, left_major_tensor and right_major_tensor
/// such deducing hint is not available in CUDA due to a bug of nvcc
///

template <class T, std::integral... Shapes>
basic_tensor(T *data, Shapes... shapes) -> basic_tensor<T, layout::right_major<sizeof...(Shapes)>>;

template <class T, std::integral... Shapes>
basic_tensor(T *data, Shapes... shapes) -> basic_tensor<T, layout::left_major<sizeof...(Shapes)>>;

template <class T, int RANK>
using left_major_tensor = basic_tensor<T, layout::left_major<RANK>>;

template <class T, int RANK>
using right_major_tensor = basic_tensor<T, layout::right_major<RANK>>;

template <class T, int RANK>
using tensor = right_major_tensor<T, RANK>;

///
/// type casting utilities
///

template <class T, class U, class Layout> requires(std::is_convertible_v<U *, T *>)
basic_tensor<T, Layout> tensor_cast(const basic_tensor<U, Layout> &tensor) {
    return basic_tensor<T, Layout>{tensor.data(), tensor.layout()};
}

template <class T, class Layout>
basic_tensor<const T, Layout> tensor_cast(const basic_tensor<T, Layout> &tensor) {
    return tensor_cast<const T, T, Layout>(tensor);
}

///
/// work-around maker functions for current cuda implementation
///

template <class T>
inline auto make_left_major_tensor(T *data, std::integral auto... shapes) {
    constexpr int rank = sizeof...(shapes);
    return basic_tensor<T, layout::left_major<rank>>{data, shapes...};
}

template <class T>
inline auto make_right_major_tensor(T *data, std::integral auto... shapes) {
    constexpr int rank = sizeof...(shapes);
    return basic_tensor<T, layout::right_major<rank>>{data, shapes...};
}

template <class T, std::integral... Shapes>
inline auto make_tensor(T *data, Shapes... shapes) {
    return make_right_major_tensor(data, shapes...);
}
} // namespace moke
