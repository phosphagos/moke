#pragma once
#include "moke/arithmetic.hpp"
#include "moke/common.hpp"
#include "moke/meta.hpp"
#include <concepts>

namespace moke {
template <int RANK>
class layout {
private:
    static_assert(RANK > 0, "layout must have more than one rank(s).");
    int32_t m_shape[RANK];
    uint32_t m_stride[RANK];

public:
    MOKE_INLINE layout() noexcept : m_shape{0}, m_stride{0} {}

    template <class IDX_T>
    MOKE_INLINE layout(array_ref<IDX_T, RANK> shape) noexcept {
        MOKE_UNROLL for (int i = 0; i < RANK; i++) { m_shape[i] = shape[i]; }
        shape_to_stride();
    }

    MOKE_INLINE layout(std::integral auto... shape) noexcept requires(sizeof...(shape) == RANK)
            : m_shape{int32_t(shape)...} { this->shape_to_stride(); }

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
    template <int IDX = 0>
    MOKE_CONSTEXPR size_t offset_of(const std::integral auto &coord) const noexcept {
        return IDX == RANK - 1 ? coord : (size_t)coord * m_stride[IDX];
    }

    template <int IDX = 0>
    MOKE_CONSTEXPR size_t offset_of(const std::integral auto &coord, const std::integral auto &...coords) const noexcept {
        return (size_t)coord * m_stride[IDX] + offset_of<IDX + 1>(coords...);
    }

    MOKE_INLINE void shape_to_stride() noexcept {
        m_stride[RANK - 1] = 1;
        MOKE_UNROLL for (int i = RANK - 2; i >= 0; i--) {
            m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
        }
    }
};

template <std::integral... Shape>
layout(Shape... shape) -> layout<sizeof...(Shape)>;
} // namespace moke
