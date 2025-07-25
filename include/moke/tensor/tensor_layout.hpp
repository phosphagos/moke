#pragma once
#include "moke/common/def.hpp"
#include <type_traits>

namespace moke {
template <size_t NDims, size_t ElemSize>
class TensorLayout {
protected:
    size_t m_shape[NDims];
    size_t m_stride[NDims];

public:
    MOKE_TRIVIAL TensorLayout() noexcept : m_shape{0}, m_stride{0} {}

    MOKE_TRIVIAL TensorLayout(const size_t *shape) {
        memcpy(this->m_shape, shape, NDims * sizeof(size_t));
        shape_to_stride();
    }

    template <class... Shape, typename = std::enable_if_t<(sizeof...(Shape) == NDims)>>
    MOKE_TRIVIAL TensorLayout(Shape &&...shape) : m_shape{shape...} { shape_to_stride(); }

    MOKE_TRIVIAL size_t rank() const { return NDims; }

    MOKE_TRIVIAL size_t size() const { return m_shape[0] * m_stride[0]; }

    MOKE_TRIVIAL size_t bytes() const { return this->size() * ElemSize; }

    MOKE_TRIVIAL size_t shape(index_t dim = 0) const { return m_shape[dim]; }

    MOKE_TRIVIAL size_t stride(index_t dim = 0) const { return m_stride[dim]; }

    MOKE_TRIVIAL bool empty() const { return m_shape[0] == 0 || m_stride[0] == 0; }

private:
    MOKE_TRIVIAL void shape_to_stride() {
        m_stride[NDims - 1] = 1;
        if constexpr (NDims > 1) {
            for (int i = NDims - 2; i >= 0; --i) {
                m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
            }
        }
    }
};
} // namespace moke
