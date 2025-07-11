#pragma once
#include "moke/common/def.hpp"

namespace moke {
template <class T, size_t NDIMS, template <class, size_t> class Iterator>
struct SubTensorIterator {
    using type = Iterator<T, NDIMS - 1>;
};

template <class T, template <class, size_t> class Iterator>
struct SubTensorIterator<T, 1, Iterator> {
    using type = T &;
};

template <size_t NDIMS>
struct IteratorLayout {
    const size_t *shape;
    const size_t *stride;

    MOKE_TRIVIAL IteratorLayout(const size_t *shape, const size_t *stride)
            : shape{shape}, stride{stride} {}

    MOKE_TRIVIAL size_t size() const { return shape[0] * stride[0]; }
};

template <class T, size_t NDIMS, template <class, size_t> class Iterator>
class TensorIteratorBase {
    static_assert(NDIMS > 0, "tensor_iterator with ndims <= 0 is not allowed");
    using IteratorType = typename SubTensorIterator<T, NDIMS, Iterator>::type;

protected:
    T *m_data;
    IteratorLayout<NDIMS> m_layout;

public:
    MOKE_TRIVIAL TensorIteratorBase(T *data, IteratorLayout<NDIMS> layout) noexcept
            : m_data{data}, m_layout{layout} {}

    MOKE_UNIFIED size_t shape(index_t dim = 0) const { return m_layout.shape[dim]; }

    MOKE_UNIFIED size_t stride(index_t dim = 0) const { return m_layout.stride[dim]; }

    MOKE_UNIFIED size_t size() const { return m_layout.size(); }

    MOKE_UNIFIED size_t bytes() const { return size() * sizeof(T); }

    MOKE_UNIFIED T *data() const { return m_data; }

    MOKE_UNIFIED T *operator&() const { return m_data; }

protected:
    MOKE_UNIFIED IteratorType operator[](index_t index) const {
        if constexpr (NDIMS > 1) {
            return {m_data + index * m_layout.stride[0], m_layout.shape + 1, m_layout.stride + 1};
        } else {
            return m_data[index];
        }
    }

    MOKE_UNIFIED IteratorType operator*() const {
        if constexpr (NDIMS > 1) {
            return {m_data, m_layout.shape + 1, m_layout.stride + 1};
        } else {
            return *m_data;
        }
    }
};

template <class T, size_t NDIMS>
class DeviceTensorIterator : TensorIteratorBase<T, NDIMS, DeviceTensorIterator> {
private:
    using Base = TensorIteratorBase<T, NDIMS, DeviceTensorIterator>;
    using Base::m_data, Base::m_layout;

public:
    template <class... LayoutArgs>
    MOKE_TRIVIAL DeviceTensorIterator(T *data, LayoutArgs &&...layout) noexcept
            : Base{data, IteratorLayout<NDIMS>{layout...}} {}

    MOKE_DEVICE size_t shape(index_t dim = 0) const { return Base::shape(dim); }

    MOKE_DEVICE size_t stride(index_t dim = 0) const { return Base::stride(dim); }

    MOKE_DEVICE size_t size() const { return Base::size(); }

    MOKE_DEVICE size_t bytes() const { return Base::bytes(); }

    MOKE_DEVICE T *data() const { return Base::data(); }

    MOKE_DEVICE T *operator&() const { return Base::operator*(); }

    MOKE_DEVICE decltype(auto) operator[](index_t index) const { return Base::operator[](index); }

    MOKE_DEVICE decltype(auto) operator*() const { return Base::operator*(); }
};

template <class T, size_t NDIMS>
class HostTensorIterator : TensorIteratorBase<T, NDIMS, HostTensorIterator> {
private:
    using Base = TensorIteratorBase<T, NDIMS, HostTensorIterator>;
    using Base::m_data, Base::m_layout;

public:
    template <class... LayoutArgs>
    MOKE_TRIVIAL HostTensorIterator(T *data, LayoutArgs &&...layout) noexcept
            : Base{data, IteratorLayout<NDIMS>{layout...}} {}

    MOKE_HOST size_t shape(index_t dim = 0) const { return Base::shape(dim); }

    MOKE_HOST size_t stride(index_t dim = 0) const { return Base::stride(dim); }

    MOKE_HOST size_t size() const { return Base::size(); }

    MOKE_HOST size_t bytes() const { return Base::bytes(); }

    MOKE_HOST T *data() const { return Base::data(); }

    MOKE_HOST T *operator&() const { return Base::operator*(); }

    MOKE_HOST decltype(auto) operator[](index_t index) const { return Base::operator[](index); }

    MOKE_HOST decltype(auto) operator*() const { return Base::operator*(); }
};
} // namespace moke
