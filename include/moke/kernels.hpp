#pragma once
#include "moke/common.hpp"
#include "moke/tensor.hpp"
#include "moke/tensor/tensor_view.hpp"

namespace moke {
template <class T>
void VectorAdd(const DeviceArrayView<T> &x, const DeviceArrayView<T> &y, DeviceArrayView<T> &out);
} // namespace moke
