#include "matmul.hpp"
#include <moke/utils.hpp>
#include <gtest/gtest.h>

template <class T>
struct MatmulBenchmark : testing::Test {};

using moke::C;
using dtypes = moke::type_tuple<float, moke::half_t, moke::bfloat16_t>;
using shapes_mn = moke::type_tuple<C<128>, C<256>, C<512>, C<1024>>;
using shapes_k = moke::type_tuple<C<1024>>;
using test_params = moke::convert_t<testing::Types, moke::product_t<dtypes, shapes_mn, shapes_k>>;

TYPED_TEST_SUITE(MatmulBenchmark, test_params);

TYPED_TEST(MatmulBenchmark, AccuVerification) {
    using dtype = moke::get_t<TypeParam, 0>;
    constexpr int M = moke::get_value<TypeParam, 1>();
    constexpr int N = moke::get_value<TypeParam, 1>();
    constexpr int K = moke::get_value<TypeParam, 2>();
    auto a_dev = moke::device_vector<dtype>(M * K) | moke::fill::random{}.seed(0);
    auto b_dev = moke::device_vector<dtype>(N * K) | moke::fill::random{}.seed(1);
    auto d_dev = moke::device_vector<dtype>(M * N) | moke::fill::constant(0.0f);
    auto d_ref = moke::device_vector<dtype>(M * N) | moke::fill::constant(0.0f);

    moke::ops::matmul_baseline<dtype>(a_dev, b_dev, d_ref, M, N, K);
    moke::ops::matmul_indexing_tensor<dtype>(a_dev, b_dev, d_dev, M, N, K);
    moke::sync_device();

    bool success = d_dev | moke::compare::all_equals(d_ref);
    EXPECT_TRUE(success);
}

static int n_fallbacks = 0;

TYPED_TEST(MatmulBenchmark, PerfBenchmark) {
    using dtype = moke::get_t<TypeParam, 0>;
    constexpr int M = moke::get_value<TypeParam, 1>();
    constexpr int N = moke::get_value<TypeParam, 1>();
    constexpr int K = moke::get_value<TypeParam, 2>();
    moke::device_profiler baseline_prof{100, 300};
    moke::device_profiler tensor_prof{100, 300};

    constexpr size_t size = M * N * K;
    auto a = moke::device_vector<dtype>(M * K) | moke::fill::random{}.seed(0);
    auto b = moke::device_vector<dtype>(N * K) | moke::fill::random{}.seed(1);
    auto d = moke::device_vector<dtype>(M * N);

    for (auto _ : baseline_prof) { moke::ops::matmul_baseline<dtype>(a, b, d, M, N, K); }
    for (auto _ : tensor_prof) { moke::ops::matmul_indexing_tensor<dtype>(a, b, d, M, N, K); }
    auto time_baseline = baseline_prof.elapsed(std::micro{});
    auto time_tensor = tensor_prof.elapsed(std::micro{});

    auto perf_baseline = baseline_prof.perf(size, std::giga{});
    auto perf_tensor = tensor_prof.perf(size, std::giga{});

    auto rate_baseline = perf_tensor / perf_baseline * 100;

    std::printf("=== matmul_benchmark (M,N,K = %d,%d,%d) ===\n", M, N, K);
    std::printf("                          %8s %8s\n", "baseline", "tensor");
    std::printf("        elapsed time(us): %8lg %8lg\n", time_baseline, time_tensor);
    std::printf("    compute perf(GFLOPS): %8lg %8lg\n", perf_baseline, perf_tensor);
    std::printf("            perf rate(%%): %8lg\n", rate_baseline);
    EXPECT_TRUE(rate_baseline >= 99.9);

    if (rate_baseline < 100) { n_fallbacks++; }
    std::printf("%d fallback(s) occurred compared with mdspan.\n", n_fallbacks);
}
