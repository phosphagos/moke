#pragma once
#include "moke/common.hpp"
#include "moke/tensor/layout.hpp"
#include <type_traits>

namespace moke {
template <class T, int RANK>
class tensor {
private:
    T *m_data;
    layout<RANK> m_layout;

public:
    template <class, int> friend class tensor;

    MOKE_INLINE tensor() noexcept : m_data{nullptr}, m_layout{} {}

    template <class IDX_T>
    MOKE_INLINE tensor(T *data, array_ref<IDX_T, RANK> shape) noexcept
            : m_data{data}, m_layout{shape} {}

    MOKE_INLINE tensor(T *data, const std::integral auto &...shape) noexcept
            : m_data{data}, m_layout{shape...} {}

    template <class U> requires(std::is_convertible_v<U *, T *>)
    MOKE_INLINE tensor(const tensor<U, RANK> &other) noexcept
            : m_data{other.m_data}, m_layout{other.m_layout} {}

    MOKE_INLINE auto rank() const noexcept { return RANK; }

    MOKE_INLINE auto size() const noexcept { return m_layout.size(); }

    MOKE_INLINE auto bytes() const noexcept { return size() * sizeof(T); }

    MOKE_INLINE auto shape(int dim) const noexcept { return m_layout.shape(dim); }

    MOKE_INLINE auto stride(int dim) const noexcept { return m_layout.stride(dim); }

    MOKE_INLINE bool empty() const noexcept { return m_data == nullptr || m_layout.empty(); }

    MOKE_INLINE T *data() const noexcept { return m_data; }

    template <std::integral... Coord> requires(sizeof...(Coord) < RANK)
    MOKE_INLINE T *operator()(const Coord &...coord) const noexcept { return m_data + m_layout(coord...); }

    template <std::integral... Coord> requires(sizeof...(Coord) == RANK)
    MOKE_INLINE T &operator()(const Coord &...coord) const noexcept { return m_data[m_layout(coord...)]; }
};

template <class T, class Idx, int RANK>
tensor(T *data, const Idx (&shape)[RANK]) -> tensor<T, RANK>;

template <class T, std::integral... Shape>
tensor(T *data, const Shape &...) -> tensor<T, sizeof...(Shape)>;
} // namespace moke
