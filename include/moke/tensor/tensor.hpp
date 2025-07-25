#pragma once
#include "moke/common/def.hpp"
#include "moke/common/memory.hpp"
#include "moke/tensor/tensor_iterator.hpp"
#include "moke/tensor/tensor_layout.hpp"
#include "moke/tensor/tensor_view.hpp"

namespace moke {
template <class T, size_t NDIMS, class Allocator>
class TensorBase : public TensorLayout<NDIMS, sizeof(T)>, private Allocator {
protected:
    using Layout = TensorLayout<NDIMS, sizeof(T)>;
    using Layout::m_shape, Layout::m_stride;
    T *m_data;

public:
    MOKE_HOST TensorBase() noexcept : Layout{}, m_data{nullptr} {}

    template <class... Shape>
    MOKE_HOST TensorBase(Shape &&...shape)
            : Layout{shape...} { Allocator::allocate((void **)&m_data, this->bytes()); }

    MOKE_HOST ~TensorBase() { Allocator::deallocate(m_data); }

    MOKE_HOST TensorBase(const TensorBase &) = delete;

    MOKE_HOST TensorBase(TensorBase &&other) noexcept : Layout{other}, m_data{other.m_data} {
        static_cast<Layout &>(other) = Layout{};
        other.m_data = nullptr;
    }

    MOKE_HOST TensorBase &operator=(const TensorBase &) = delete;

    MOKE_HOST TensorBase &operator=(TensorBase &&other) noexcept {
        std::swap(m_shape, other.m_shape);
        std::swap(m_data, other.m_data);
        return *this;
    }

    MOKE_HOST T *data() const { return this->m_data; }

    template <template <class, size_t> class OtherTensor>
    MOKE_HOST void load(const OtherTensor<T, NDIMS> &other) const {
        Memcpy(this->data(), other.data(), this->bytes());
    }

    template <template <class, size_t> class OtherTensor>
    MOKE_HOST void store(OtherTensor<T, NDIMS> &other) const {
        Memcpy(other.data(), this->data(), this->bytes());
    }
};

template <class T, size_t NDIMS>
class HostTensor : public TensorBase<T, NDIMS, HostAllocator> {
private:
    using IteratorType = typename SubTensorIterator<T, NDIMS, HostTensorIterator>::type;
    using Base = TensorBase<T, NDIMS, HostAllocator>;
    using Layout = TensorLayout<NDIMS, sizeof(T)>;
    using Base::m_data, Layout::m_shape, Layout::m_stride;

public:
    using TensorBase<T, NDIMS, HostAllocator>::TensorBase;

    MOKE_HOST IteratorType operator[](index_t index) const {
        if constexpr (NDIMS > 1) {
            return {this->m_data + index * this->m_stride[0],
                    this->m_shape + 1, this->m_stride + 1};
        } else {
            return this->m_data[index];
        }
    }

    MOKE_HOST IteratorType operator*() const {
        if constexpr (NDIMS > 1) {
            return {this->m_data, this->m_shape + 1, this->m_stride + 1};
        } else {
            return *this->m_data;
        }
    }

    MOKE_HOST operator HostTensorView<T, NDIMS>() const {
        return {m_data, static_cast<const Layout &>(*this)};
    }

    MOKE_HOST operator HostTensorView<const T, NDIMS>() const {
        return {m_data, static_cast<const Layout &>(*this)};
    }
};

/// @note The data of device tensor is not accessible from host.
///       Hence, it is not indexable and required to be accessed from the device via DeviceTensorView.
template <class T, size_t NDIMS>
class DeviceTensor : public TensorBase<T, NDIMS, DeviceAllocator> {
private:
    using Base = TensorBase<T, NDIMS, DeviceAllocator>;
    using Layout = TensorLayout<NDIMS, sizeof(T)>;
    using Base::m_data, Layout::m_shape, Layout::m_stride;

public:
    using TensorBase<T, NDIMS, DeviceAllocator>::TensorBase;

    MOKE_HOST operator DeviceTensorView<T, NDIMS>() const {
        return {m_data, static_cast<const Layout &>(*this)};
    }

    MOKE_HOST operator DeviceTensorView<const T, NDIMS>() const {
        return {m_data, static_cast<const Layout &>(*this)};
    }
};

template <class T> using HostArray = HostTensor<T, 1>;
template <class T> using DeviceArray = DeviceTensor<T, 1>;
} // namespace moke
