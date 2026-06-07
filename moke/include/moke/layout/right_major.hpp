#pragma once
#include "moke/arithmetic.hpp"
#include "moke/common.hpp"
#include "moke/meta.hpp"

namespace moke::layout {
template <int RANK_>
struct right_major {
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
    MOKE_CONSTEXPR right_major() noexcept : m_shape{0}, m_stride{0} {}

    template <std::integral... Shape> requires(sizeof...(Shape) == RANK)
    MOKE_CONSTEXPR right_major(Shape ...shape) noexcept
            : m_shape{shape_t(shape)...} { shape_to_stride(); }

    MOKE_CONSTEXPR void shape_to_stride() {
        stride_t stride = 1;
        for (int i = RANK - 1; i >= 0; i--) {
            m_stride[i] = stride;
            stride *= m_shape[i];
        }
    }

    MOKE_INLINE size_t size() const noexcept { return m_shape[0] * m_stride[0]; }

    MOKE_INLINE bool empty() const noexcept { return m_shape[0] == 0 || m_stride[0] == 0; }

    MOKE_INLINE size_t operator()(std::integral auto... coord) const noexcept {
        static_assert((sizeof...(coord) <= RANK));
        return offset_of(coord...);
    }

    template <int IDX = 0>
    MOKE_CONSTEXPR size_t offset_of(const std::integral auto &coord) const noexcept {
        return IDX == RANK - 1 ? coord : (size_t)coord * m_stride[IDX];
    }

    template <int IDX = 0>
    MOKE_CONSTEXPR size_t offset_of(const std::integral auto &coord, const std::integral auto &...coords) const noexcept {
        return (size_t)coord * m_stride[IDX] + offset_of<IDX + 1>(coords...);
    }
};
} // namespace moke
