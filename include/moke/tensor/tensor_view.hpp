#pragma once
#include "moke/common/def.hpp"
#include "moke/tensor/tensor_iterator.hpp"
#include "moke/tensor/tensor_layout.hpp"

namespace moke {
template <class T, size_t NDIMS, template <class, size_t> class Iterator>
class TensorViewBase : public TensorLayout<NDIMS, sizeof(T)> {
protected:
    using IteratorType = typename SubTensorIterator<T, NDIMS, Iterator>::type;
    using Layout = TensorLayout<NDIMS, sizeof(T)>;
    T *m_data;

public:
    MOKE_TRIVIAL TensorViewBase() noexcept : Layout{}, m_data{nullptr} {}

    template <class... LayoutArgs>
    MOKE_TRIVIAL TensorViewBase(T *data, LayoutArgs &&...layout) noexcept
            : Layout{layout...}, m_data{data} {}

    MOKE_UNIFIED T *data() const { return this->m_data; }

    MOKE_UNIFIED IteratorType operator[](index_t index) const {
        if constexpr (NDIMS > 1) {
            return {this->m_data + index * this->m_stride[0],
                    this->m_shape + 1, this->m_stride + 1};
        } else {
            return this->m_data[index];
        }
    }

    MOKE_UNIFIED IteratorType operator*() const {
        if constexpr (NDIMS > 1) {
            return {this->m_data, this->m_shape + 1, this->m_stride + 1};
        } else {
            return *this->m_data;
        }
    }

    template <template <class, size_t> class OtherTensor>
    MOKE_HOST void load(const OtherTensor<T, NDIMS> &other) const {
        Memcpy(this->data(), other.data(), this->bytes());
    }

    template <template <class, size_t> class OtherTensor>
    MOKE_HOST void store(OtherTensor<T, NDIMS> &other) const {
        Memcpy(other.m_data, this->data(), this->bytes());
    }
};

template <class T, size_t NDIMS>
class DeviceTensorView : public TensorViewBase<T, NDIMS, DeviceTensorIterator> {
private:
    using Base = TensorViewBase<T, NDIMS, DeviceTensorIterator>;
    using Layout = typename Base::Layout;
    using Base::m_data, Layout::m_shape, Layout::m_stride;

public:
    MOKE_TRIVIAL DeviceTensorView() noexcept : Base{} {}

    template <class... Shape>
    MOKE_TRIVIAL DeviceTensorView(T *data, Shape &&...shape) noexcept : Base{data, shape...} {}

    MOKE_DEVICE decltype(auto) operator[](index_t index) const { return Base::operator[](index); }

    MOKE_DEVICE decltype(auto) operator*() const { return Base::operator*(); }
};

template <class T, size_t NDIMS>
class HostTensorView : public TensorViewBase<T, NDIMS, HostTensorIterator> {
private:
    using Base = TensorViewBase<T, NDIMS, HostTensorIterator>;
    using Layout = typename Base::Layout;
    using Base::m_data, Layout::m_shape, Layout::m_stride;

public:
    MOKE_TRIVIAL HostTensorView() noexcept : Base{} {}

    template <class... Shape>
    MOKE_TRIVIAL HostTensorView(T *data, Shape &&...shape) noexcept : Base{data, shape...} {}

    MOKE_HOST decltype(auto) operator[](index_t index) const { return Base::operator[](index); }

    MOKE_HOST decltype(auto) operator*() const { return Base::operator*(); }
};

template <class T> using HostArrayView = HostTensorView<T, 1>;
template <class T> using DeviceArrayView = DeviceTensorView<T, 1>;
} // namespace moke
