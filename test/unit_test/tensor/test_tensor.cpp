#include <moke/mokeutils.hpp>
#include <moke/tensor.hpp>
#include <gtest/gtest.h>

TEST(TestTensor, TestMatmul) {
    constexpr size_t B = 4, M = 32, N = 16, K = 1024;
    auto vec_a = moke::host_vector<float>{B * M * K} | moke::fill::random{}.seed(0);
    auto vec_b = moke::host_vector<float>{B * N * K} | moke::fill::random{}.seed(1);
    auto vec_d = moke::host_vector<float>{B * M * N} | moke::fill::constant{0};
    auto ref_d = moke::host_vector<float>{B * M * N} | moke::fill::constant{0};

    for (int i = 0; i < B; i++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    ref_d[i * M * N + m * N + n] += vec_a[i * M * K + m * K + k] * vec_b[i * N * K + n * K + k];
                }
            }
        }
    }

    moke::tensor<float, 3> a{vec_a, {B, M, K}};
    moke::tensor<float, 3> b{vec_b, {B, N, K}};
    moke::tensor<float, 3> d{vec_d, {B, M, N}};
    for (int i = 0; i < B; i++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    d(i, m, n) += a(i, m, k) * b(i, n, k);
                }
            }
        }
    }

    EXPECT_TRUE(vec_d | moke::compare::all_equals{ref_d});
}
