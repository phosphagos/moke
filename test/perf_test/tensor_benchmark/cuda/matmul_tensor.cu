#include "moke/tensor.hpp"
#include "matmul.hpp"

namespace moke::ops::kernel {
template <class dtype>
__global__ void matmul_indexing_tensor(tensor<const dtype, 2> a, tensor<const dtype, 2> b, tensor<dtype, 2> d) {
    const auto M = d.shape(0);
    const auto N = d.shape(1);
    const auto K = a.shape(1);

    const auto m = blockIdx.y * blockDim.y + threadIdx.y;
    const auto n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) { return; }

    dtype accumulate = 0;
    for (int k = 0; k < K; k++) {
        accumulate += dtype(a(m, k) * b(k, n));
    }
    d(m, n) = accumulate;
}
} // namespace moke::ops::kernel

namespace moke::ops {
template <class dtype>
void matmul_indexing_tensor(const dtype *a, const dtype *b, dtype *d, int M, int N, int K) {
    const dim3 nthreads{32, 32};
    const dim3 nblocks{unsigned(N + 31) / 32, unsigned(M + 31) / 32};
    kernel::matmul_indexing_tensor<<<nblocks, nthreads>>>(tensor{a, M, K}, tensor{b, K, N}, tensor{d, M, N});
}

template void matmul_indexing_tensor(const float *, const float *, float *, int, int, int);
template void matmul_indexing_tensor(const half_t *, const half_t *, half_t *, int, int, int);
template void matmul_indexing_tensor(const bfloat16_t *, const bfloat16_t *, bfloat16_t *, int, int, int);
} // namespace moke::ops
