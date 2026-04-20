#pragma once
#include "moke/common.hpp"
#include "moke/dtype.hpp"
#include "moke/type_traits.hpp"
#include <concepts>

namespace moke {
template <class T, size_t N>
using array_ref = const T (&)[N];

template <int RANK>
class layout {
private:
    static_assert(RANK > 0, "layout must have more than one rank(s).");
    int m_shape[RANK];
    int m_stride[RANK];

public:
    MOKE_INLINE layout() noexcept : m_shape{0}, m_stride{0} {}

    MOKE_INLINE layout(array_ref<int, RANK> shape) noexcept {
        MOKE_UNROLL for (int i = 0; i < RANK; i++) { m_shape[i] = shape[i]; }
        shape_to_stride();
    }

    MOKE_INLINE layout(std::integral auto... shape) noexcept requires(sizeof...(shape) == RANK)
            : m_shape{shape...} { this->shape_to_stride(); }

    template <int R> requires(R < RANK)
    MOKE_INLINE size_t operator()(array_ref<int, R> coord) const noexcept;

    MOKE_INLINE size_t operator()(array_ref<int, RANK> coord) const noexcept;

    MOKE_INLINE size_t operator()(std::integral auto... coord) const noexcept {
        static_assert((sizeof...(coord) <= RANK));
        return offset_of(coord...);
    }

    MOKE_INLINE size_t size() const noexcept { return m_shape[0] * m_stride[0]; }

    MOKE_INLINE int rank() const noexcept { return RANK; }

    MOKE_INLINE int shape(int dim) const noexcept { return m_shape[dim]; }

    MOKE_INLINE int stride(int dim) const noexcept { return m_stride[dim]; }

    MOKE_INLINE bool empty() const noexcept { return m_shape[0] == 0 || m_stride[0] == 0; }

private:
    // compute stride of continuous layout with given shape
    MOKE_INLINE void shape_to_stride() noexcept;

    template <int IDX = 0>
    MOKE_CONSTEXPR size_t offset_of(const std::integral auto &coord) const noexcept {
        return IDX == RANK - 1 ? coord : coord * m_stride[IDX];
    }

    template <int IDX = 0>
    MOKE_CONSTEXPR size_t offset_of(const std::integral auto &coord, const std::integral auto &...coords) const noexcept {
        return coord * m_stride[IDX] + offset_of<IDX + 1>(coords...);
    }
};

template <std::integral... Shape>
layout(Shape... shape) -> layout<sizeof...(Shape)>;

template <int RANK> template <int R> requires(R < RANK)
MOKE_INLINE size_t layout<RANK>::operator()(array_ref<int, R> coord) const noexcept {
    size_t offset = 0;
    MOKE_UNROLL for (int i = 0; i < R; i++) {
        offset += m_stride[i] * coord[i];
    }
    return offset;
}

template <int RANK>
MOKE_INLINE size_t layout<RANK>::operator()(array_ref<int, RANK> coord) const noexcept {
    size_t offset = coord[RANK - 1];
    MOKE_UNROLL for (int i = 0; i < RANK - 1; i++) {
        offset += m_stride[i] * coord[i];
    }
    return offset;
}

template <int RANK>
MOKE_INLINE void layout<RANK>::shape_to_stride() noexcept {
    m_stride[RANK - 1] = 1;
    MOKE_UNROLL for (int i = RANK - 2; i >= 0; i--) {
        m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
    }
}
} // namespace moke
