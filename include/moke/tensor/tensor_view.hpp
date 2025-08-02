#pragma once
#include "moke/common/def.hpp"
#include "moke/tensor/tensor_iterator.hpp"
#include <type_traits>

namespace moke {
template <size_t NDims>
struct TensorLayout {
    size_t shape[NDims];
    size_t stride[NDims];

    MOKE_TRIVIAL TensorLayout() noexcept : shape{0}, stride{0} {}

    MOKE_TRIVIAL TensorLayout(const size_t *shape) {
        memcpy(this->shape, shape, NDims * sizeof(size_t));
        shape_to_stride();
    }

    template <class... Shape, typename = std::enable_if_t<(sizeof...(Shape) == NDims)>>
    MOKE_TRIVIAL TensorLayout(Shape &&...shape) : shape{shape...} { shape_to_stride(); }

    MOKE_TRIVIAL size_t size() const { return shape[0] * stride[0]; }

private:
    MOKE_TRIVIAL void shape_to_stride() {
        stride[NDims - 1] = 1;
        if constexpr (NDims > 1) {
            for (int i = NDims - 2; i >= 0; --i) {
                stride[i] = stride[i + 1] * shape[i + 1];
            }
        }
    }
};

template <class T, size_t NDIMS> class DeviceTensor;
template <class T, size_t NDIMS> class HostTensor;

template <class T, size_t NDIMS, template <class, size_t> class Iterator>
class TensorViewBase {
protected:
    T *m_data;
    TensorLayout<NDIMS> m_layout;

public:
    using IteratorType = typename SubTensorIterator<T, NDIMS, Iterator>::type;

    using ConstIteratorType = typename SubTensorIterator<const T, NDIMS, Iterator>::type;

    MOKE_TRIVIAL TensorViewBase() noexcept : m_data{nullptr}, m_layout{} {}

    template <class... LayoutArgs>
    MOKE_TRIVIAL TensorViewBase(T *data, LayoutArgs &&...layout) noexcept
            : m_data{data}, m_layout{layout...} {}

    MOKE_TRIVIAL size_t rank() const { return NDIMS; }

    MOKE_TRIVIAL size_t size() const { return m_layout.size(); }

    MOKE_TRIVIAL size_t bytes() const { return this->size() * sizeof(T); }

    MOKE_TRIVIAL size_t shape(index_t dim = 0) const { return m_layout.shape[dim]; }

    MOKE_TRIVIAL size_t stride(index_t dim = 0) const { return m_layout.stride[dim]; }

    MOKE_TRIVIAL bool empty() const {
        return m_layout.shape[0] == 0 || m_layout.stride[0] == 0;
    }

    MOKE_UNIFIED T *data() const { return this->m_data; }

    template <template <class, size_t> class OtherTensor>
    MOKE_HOST void load(const OtherTensor<T, NDIMS> &other) const {
        Memcpy(this->data(), other.data(), this->bytes());
    }

    template <template <class, size_t> class OtherTensor>
    MOKE_HOST void store(OtherTensor<T, NDIMS> &other) const {
        Memcpy(other.data(), this->data(), this->bytes());
    }

    MOKE_UNIFIED ConstIteratorType operator[](index_t index) const {
        if constexpr (NDIMS > 1) {
            return {this->m_data + index * m_layout.stride[0],
                    m_layout.shape + 1, m_layout.stride + 1};
        } else {
            return this->m_data[index];
        }
    }

    MOKE_UNIFIED IteratorType operator[](index_t index) {
        if constexpr (NDIMS > 1) {
            return {this->m_data + index * m_layout.stride[0],
                    m_layout.shape + 1, m_layout.stride + 1};
        } else {
            return this->m_data[index];
        }
    }

    MOKE_UNIFIED ConstIteratorType operator*() const {
        if constexpr (NDIMS > 1) {
            return {this->m_data, m_layout.shape + 1, m_layout.stride + 1};
        } else {
            return *this->m_data;
        }
    }

    MOKE_UNIFIED IteratorType operator*() {
        if constexpr (NDIMS > 1) {
            return {this->m_data, m_layout.shape + 1, m_layout.stride + 1};
        } else {
            return *this->m_data;
        }
    }
};

template <class T, size_t NDIMS>
class DeviceTensorView : public TensorViewBase<T, NDIMS, DeviceTensorIterator> {
private:
    using Base = TensorViewBase<T, NDIMS, DeviceTensorIterator>;
    using Base::m_data, Base::m_layout;
    friend class DeviceTensor<T, NDIMS>;

public:
    MOKE_TRIVIAL DeviceTensorView() noexcept : Base{} {}

    template <class... Shape>
    MOKE_TRIVIAL DeviceTensorView(T *data, Shape &&...shape) noexcept : Base{data, shape...} {}

    MOKE_DEVICE decltype(auto) operator[](index_t index) const { return Base::operator[](index); }

    MOKE_DEVICE decltype(auto) operator[](index_t index) { return Base::operator[](index); }

    MOKE_DEVICE decltype(auto) operator*() const { return Base::operator*(); }

    MOKE_DEVICE decltype(auto) operator*() { return Base::operator*(); }
};

template <class T, size_t NDIMS>
class HostTensorView : public TensorViewBase<T, NDIMS, HostTensorIterator> {
private:
    using Base = TensorViewBase<T, NDIMS, HostTensorIterator>;
    using Base::m_data, Base::m_layout;
    friend class HostTensor<T, NDIMS>;

public:
    MOKE_TRIVIAL HostTensorView() noexcept : Base{} {}

    template <class... Shape>
    MOKE_TRIVIAL HostTensorView(T *data, Shape &&...shape) noexcept : Base{data, shape...} {}

    MOKE_HOST decltype(auto) operator[](index_t index) const { return Base::operator[](index); }

    MOKE_DEVICE decltype(auto) operator[](index_t index) { return Base::operator[](index); }

    MOKE_HOST decltype(auto) operator*() const { return Base::operator*(); }

    MOKE_DEVICE decltype(auto) operator*() { return Base::operator*(); }
};

template <class T> using HostArrayView = HostTensorView<T, 1>;
template <class T> using DeviceArrayView = DeviceTensorView<T, 1>;
} // namespace moke
