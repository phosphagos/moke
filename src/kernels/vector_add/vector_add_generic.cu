#include "moke/kernels.hpp"
#include <cstdio>

namespace moke {
template <class T>
__global__ void vector_add(const T *x, const T *y, T *out, size_t length) {
    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) { out[id] = x[id] + y[id]; }
}

template <class T>
void VectorAdd(const DeviceArrayView<T> &x, const DeviceArrayView<T> &y, DeviceArrayView<T> &out) {
    if (x.size() != y.size() || x.size() != out.size()) {
        std::printf("Error: size of x, y and output does not match.\n");
        return;
    }

    size_t len = x.size();
    int thread_dim = 1024;
    vector_add<<<ceil_div(len, thread_dim), thread_dim>>>(x.data(), y.data(), out.data(), len);
}

template void VectorAdd<float>(const DeviceArrayView<float> &, const DeviceArrayView<float> &, DeviceArrayView<float> &);
template void VectorAdd<double>(const DeviceArrayView<double> &, const DeviceArrayView<double> &, DeviceArrayView<double> &);
template void VectorAdd<int32_t>(const DeviceArrayView<int32_t> &, const DeviceArrayView<int32_t> &, DeviceArrayView<int32_t> &);
} // namespace moke
