#include "matmul.hpp"

namespace moke::ops::kernel {
template <class dtype>
__global__ void matmul_baseline(const dtype *a, const dtype *b, dtype *d, int M, int N, int K) {
    const auto m = blockIdx.y * blockDim.y + threadIdx.y;
    const auto n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) { return; }

    dtype accumulate = 0;
    for (int k = 0; k < K; k++) {
        accumulate += dtype(a[m * K + k] * b[n * K + k]);
    }
    d[m * N + n] = accumulate;
}
} // namespace moke::ops::kernel

namespace moke::ops {
template <class dtype>
void matmul_baseline(const dtype *a, const dtype *b, dtype *d, int M, int N, int K) {
    const dim3 nthreads{32, 32};
    const dim3 nblocks{unsigned(N + 31) / 32, unsigned(M + 31) / 32};
    kernel::matmul_baseline<<<nblocks, nthreads>>>(a, b, d, M, N, K);
}

template void matmul_baseline(const float *, const float *, float *, int, int, int);
template void matmul_baseline(const half_t *, const half_t *, half_t *, int, int, int);
template void matmul_baseline(const bfloat16_t *, const bfloat16_t *, bfloat16_t *, int, int, int);
} // namespace moke::ops
