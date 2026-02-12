#include "moke/native.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

namespace moke {
template <class T>
MOKE_DEVICE T random_uniform(curandState_t &state, float alpha, float beta) {
    float res = curand_uniform(&state);
    return T(res * alpha + beta);
}

template <>
MOKE_DEVICE double random_uniform<double>(curandState_t &state, float alpha, float beta) {
    double res = curand_uniform_double(&state);
    return res * alpha + beta;
}

template <class T>
MOKE_KERNEL void fill_random_kernel(T *dest, size_t length, float min, float max, uint32_t seed) {
    curandState_t state;

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto tno = blockDim.x * gridDim.x;
    float alpha = max - min;
    float beta = min;

    if (tid < length) { curand_init(seed, tid, 0, &state); }
    for (auto i = tid; i < length; i += tno) {
        dest[tid] = random_uniform<T>(state, alpha, beta);
    }
}

template <class T>
void fill_random(device_memory_t, T *dest, size_t length, float min, float max, uint32_t seed) {
    constexpr int nthreads = 1024;
    constexpr int nblocks = 256;
    fill_random_kernel<<<nthreads, nblocks>>>(dest, length, min, max, seed);
}

template <class T>
MOKE_DEVICE T random_uniform(curandState_t &state, float alpha, float beta, uint8_t bits) {
    float res = curand_uniform(&state);
    res = std::round(res * (1 << bits)) / (1 << bits);
    return res;
}

template <>
MOKE_DEVICE double random_uniform<double>(curandState_t &state, float alpha, float beta, uint8_t bits) {
    double res = curand_uniform(&state);
    res = std::round(res * (1 << bits)) / (1 << bits);
    return res;
}

template <class T>
MOKE_KERNEL void fill_random_kernel(T *dest, size_t length, float min, float max, uint32_t seed, uint8_t bits) {
    curandState_t state;

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto tno = blockDim.x * gridDim.x;
    float alpha = max - min;
    float beta = min;

    if (tid < length) { curand_init(seed, tid, 0, &state); }
    for (auto i = tid; i < length; i += tno) {
        auto rnd = random_uniform<T>(state, alpha, beta, bits);
        dest[tid] = rnd;
    }
}

template <class T>
void fill_random(device_memory_t, T *dest, size_t length, float min, float max, uint32_t seed, uint8_t bits) {
    constexpr int nthreads = 1024;
    constexpr int nblocks = 256;
    fill_random_kernel<<<nthreads, nblocks>>>(dest, length, min, max, seed, bits);
}

#define DEVICE_FILL_RANDOM(T)                                                                                \
    template void fill_random(device_memory_t, T *dest, size_t length, float min, float max, uint32_t seed); \
    template void fill_random(device_memory_t, T *dest, size_t length, float min, float max, uint32_t seed, uint8_t bits);

DEVICE_FILL_RANDOM(float);
DEVICE_FILL_RANDOM(double);
DEVICE_FILL_RANDOM(__half);
DEVICE_FILL_RANDOM(__nv_bfloat16);
} // namespace moke
