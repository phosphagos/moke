#include "moke/ops.hpp"
#include <cstdio>

namespace moke {
template <class T>
__global__ void vector_add(const T *x, const T *y, T *out, size_t length) {
    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) { out[id] = x[id] + y[id]; }
}

template <class T>
void VectorAdd(DeviceArrayView<const T> x, DeviceArrayView<const T> y, DeviceArrayView<T> out) {
    if (x.size() != y.size() || x.size() != out.size()) {
        std::printf("Error: size of x, y and output does not match.\n");
        return;
    }

    size_t len = x.size();
    int thread_dim = 1024;
    vector_add<<<ceil_div(len, thread_dim), thread_dim>>>(x.data(), y.data(), out.data(), len);
}

template void VectorAdd(DeviceArrayView<const float> x, DeviceArrayView<const float> y, DeviceArrayView<float> out);
} // namespace moke
