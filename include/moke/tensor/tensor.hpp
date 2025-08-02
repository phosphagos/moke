#pragma once
#include "moke/common/def.hpp"
#include "moke/common/memory.hpp"
#include "moke/tensor/tensor_view.hpp"
#include <algorithm>

namespace moke {
template <class T, size_t NDIMS>
class DeviceTensor : private DeviceAllocator, public DeviceTensorView<T, NDIMS> {
public:
    using Allocator = DeviceAllocator;
    using TensorView = DeviceTensorView<T, NDIMS>;

private:
    using TensorView::m_data, TensorView::m_layout;

public:
    MOKE_HOST DeviceTensor() noexcept : TensorView{nullptr} {}

    template <class... Shape>
    MOKE_HOST DeviceTensor(Shape &&...shape) : TensorView{nullptr, shape...} {
        Allocator::allocate((void **)&m_data, this->bytes());
    }

    MOKE_HOST ~DeviceTensor() { Allocator::deallocate(m_data); }

    MOKE_HOST DeviceTensor(const DeviceTensor &) = delete;

    MOKE_HOST DeviceTensor &operator=(const DeviceTensor &) = delete;

    MOKE_HOST DeviceTensor(DeviceTensor &&other) noexcept : DeviceTensor{} {
        *this = std::move(other);
    }

    MOKE_HOST DeviceTensor &operator=(DeviceTensor &&other) noexcept {
        std::swap(m_data, other.m_data);
        std::swap(m_layout, other.m_shape);
        return *this;
    }
};

template <class T, size_t NDIMS>
class HostTensor : private HostAllocator, public HostTensorView<T, NDIMS> {
public:
    using Allocator = HostAllocator;
    using TensorView = HostTensorView<T, NDIMS>;

private:
    using TensorView::m_data, TensorView::m_layout;

public:
    MOKE_HOST HostTensor() noexcept : TensorView{nullptr} {}

    template <class... Shape>
    MOKE_HOST HostTensor(Shape &&...shape) : TensorView{nullptr, shape...} {
        Allocator::allocate((void **)&m_data, this->bytes());
    }

    MOKE_HOST ~HostTensor() { Allocator::deallocate(m_data); }

    MOKE_HOST HostTensor(const HostTensor &) = delete;

    MOKE_HOST HostTensor &operator=(const HostTensor &) = delete;

    MOKE_HOST HostTensor(HostTensor &&other) noexcept : HostTensor{} {
        *this = std::move(other);
    }

    MOKE_HOST HostTensor &operator=(HostTensor &&other) noexcept {
        std::swap(m_data, other.m_data);
        std::swap(m_layout, other.m_shape);
        return *this;
    }
};

template <class T> using DeviceArray = DeviceTensor<T, 1>;
template <class T> using HostArray = HostTensor<T, 1>;
} // namespace moke
