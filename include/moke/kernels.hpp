#pragma once
#include "moke/common.hpp"
#include "moke/tensor.hpp"
#include "moke/tensor/tensor_view.hpp"

namespace moke {
template <class T>
void VectorAdd(DeviceArrayView<const T> x, DeviceArrayView<const T> y, DeviceArrayView<T> out);
} // namespace moke
