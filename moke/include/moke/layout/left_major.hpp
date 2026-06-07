#pragma once
#include "moke/arithmetic.hpp"
#include "moke/common.hpp"
#include "moke/meta.hpp"

namespace moke::layout {
template <int RANK_>
struct left_major {
// definations:
    constexpr static int RANK = RANK_;
    static_assert(RANK > 0);
    using shape_t = int32_t;
    using stride_t = uint32_t;
    using offset_t = size_t;

// data members:
    shape_t m_shape[RANK];
    stride_t m_stride[RANK];

// methods:
    MOKE_CONSTEXPR left_major() noexcept : m_shape{0}, m_stride{0} {}

    template <std::integral... Shape> requires(sizeof...(Shape) == RANK)
    MOKE_CONSTEXPR left_major(Shape ...shape) noexcept
            : m_shape{shape_t(shape)...} { shape_to_stride(); }

    MOKE_CONSTEXPR void shape_to_stride() {
        stride_t stride = 1;
        MOKE_UNROLL for (int i = 0; i < RANK; i++) {
            m_stride[i] = stride;
            stride *= m_shape[i];
        }
    }

    MOKE_INLINE size_t size() const noexcept { return m_shape[RANK - 1] * m_stride[RANK - 1]; }

    MOKE_INLINE bool empty() const noexcept { return m_shape[RANK - 1] == 0 || m_stride[RANK - 1] == 0; }

    MOKE_INLINE offset_t operator()(std::integral auto coord, std::integral auto... coords) const noexcept {
        static_assert((sizeof...(coords) < RANK));
        return coord + offset_of(coords...);
    }

    template <int IDX = 1>
    MOKE_CONSTEXPR offset_t offset_of(std::integral auto coord, std::integral auto... coords) const noexcept {
        return (offset_t)coord * m_stride[IDX] + offset_of<IDX + 1>(coords...);
    }

    template <int IDX = 1>
    MOKE_CONSTEXPR offset_t offset_of(std::integral auto coord) const noexcept {
        return (offset_t)coord * m_stride[IDX];
    }
};
} // namespace moke
