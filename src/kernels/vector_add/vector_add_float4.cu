#include "moke/common/arithmetic.hpp"
#include "moke/kernels.hpp"
#include <cstdio>
#include <vector_functions.h>

namespace moke {
__global__ void vector_add(const float4 *x, const float4 *y, float4 *out, size_t length) {
    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id * 4 >= length) { return; }

    float4 vx = x[id];
    float4 vy = y[id];
    float4 vout = make_float4(vx.x + vy.x, vx.y + vy.y, vx.z + vy.z, vx.w + vy.w);
    out[id] = vout;
}

template <>
void VectorAdd(const DeviceArrayView<float> &x, const DeviceArrayView<float> &y, DeviceArrayView<float> &out) {
    if (x.size() != y.size() || x.size() != out.size()) {
        std::printf("Error: size of x, y and output does not match.\n");
        return;
    }

    int len = ceil_div(x.size(), 4);
    int thread_dim = 1024;
    vector_add<<<ceil_div(len, thread_dim), thread_dim>>>(
            (float4 *)x.data(), (float4 *)y.data(),
            (float4 *)out.data(), x.size()
    );
}
} // namespace moke
