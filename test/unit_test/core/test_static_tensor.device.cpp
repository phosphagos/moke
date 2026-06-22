#include <moke/common.hpp>
#include <moke/meta.hpp>
#include <moke/static_tensor.hpp>
#include <moke/tensor.hpp>
#include <moke/utils.hpp>

using moke::C;
using namespace moke::literals;

template <class TensorA, class TensorB, class TensorD, class acc_t = float>
__global__ void matmul_kernel(TensorA a, TensorB b, TensorD d) {
    const auto M = d.shape(0_ic);
    const auto N = d.shape(1_ic);
    const auto K = a.shape(1_ic);

    const auto m = blockIdx.y * blockDim.y + threadIdx.y;
    const auto n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) { return; }

    acc_t accumulate = 0;
    for (int k = 0; k < K; k++) {
        accumulate += acc_t(a(m, k) * b(k, n));
    }
    d(m, n) = accumulate;
}

template <class dtype, std::integral auto M, std::integral auto N, std::integral auto K>
void matmul_static(const dtype *a, const dtype *b, dtype *d, C<M> m, C<N> n, C<K> k) {
    const dim3 nthreads{32, 32};
    const dim3 nblocks{unsigned(N + 31) / 32, unsigned(M + 31) / 32};

    auto tA = moke::make_tensor(a, m, k);
    auto tB = moke::make_tensor(b, k, n);
    auto tD = moke::make_tensor(d, m, n);
    matmul_kernel<<<nblocks, nthreads>>>(tA, tB, tD);

    static_assert(std::same_as<decltype(tA), moke::static_tensor<const dtype,                                    //
                                                                 moke::static_layout<moke::constant_tuple<M, K>, //
                                                                                     moke::constant_tuple<K, 1>, //
                                                                                     M * K, 1>>>);
    static_assert(std::same_as<decltype(tB), moke::static_tensor<const dtype,                                    //
                                                                 moke::static_layout<moke::constant_tuple<K, N>, //
                                                                                     moke::constant_tuple<N, 1>, //
                                                                                     N * K, 1>>>);
    static_assert(std::same_as<decltype(tD), moke::static_tensor<dtype,                                          //
                                                                 moke::static_layout<moke::constant_tuple<M, N>, //
                                                                                     moke::constant_tuple<N, 1>, //
                                                                                     M * N, 1>>>);
}

template <class dtype>
void matmul_dynamic(const dtype *a, const dtype *b, dtype *d, int m, int n, int k) {
    const dim3 nthreads{32, 32};
    const dim3 nblocks{unsigned(n + 31) / 32, unsigned(m + 31) / 32};
    auto tA = moke::make_tensor(a, m, k);
    auto tB = moke::make_tensor(b, k, n);
    auto tD = moke::make_tensor(d, m, n);
    matmul_kernel<<<nblocks, nthreads>>>(tA, tB, tD);

    static_assert(std::same_as<decltype(tA), moke::tensor<const dtype, 2>>);
    static_assert(std::same_as<decltype(tB), moke::tensor<const dtype, 2>>);
    static_assert(std::same_as<decltype(tD), moke::tensor<dtype, 2>>);
}

#include <gtest/gtest.h>

TEST(TestStaticTensor, TestStaticTensorMatmul) {
    using dtype = float;
    constexpr auto M = 128_ic;
    constexpr auto N = 128_ic;
    constexpr auto K = 1024_ic;

    auto a_dev = moke::device_vector<dtype>(M * K) | moke::fill::random{}.seed(0);
    auto b_dev = moke::device_vector<dtype>(N * K) | moke::fill::random{}.seed(1);
    auto d_dev = moke::device_vector<dtype>(M * N) | moke::fill::constant(0.0f);
    auto d_ref = moke::device_vector<dtype>(M * N) | moke::fill::constant(0.0f);

    matmul_static(a_dev.data(), b_dev.data(), d_dev.data(), M, N, K);
    matmul_dynamic(a_dev.data(), b_dev.data(), d_ref.data(), M, N, K);
    moke::sync_device();

    bool success = d_dev | moke::compare::all_equals(d_ref);
    EXPECT_TRUE(success);
}
