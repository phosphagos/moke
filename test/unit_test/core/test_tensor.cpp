#include <moke/tensor.hpp>
#include <gtest/gtest.h>
#include <type_traits>

TEST(TestTensor, TestTensorConstruction) {
    int *buffer = nullptr;

    moke::tensor tensor_2d{buffer, 16, 32};
    moke::tensor tensor_3d{buffer, 16, 32u, 64l};
    moke::tensor tensor_4d{buffer, 16, 32u, 64l, 128ull};
    EXPECT_EQ(tensor_2d.rank(), 2);
    EXPECT_EQ(tensor_3d.rank(), 3);
    EXPECT_EQ(tensor_4d.rank(), 4);

    moke::tensor tensor2d{buffer, {16, 32}};
    moke::tensor tensor3d{buffer, {16, 32, 64}};
    moke::tensor tensor4d{buffer, {16, 32, 64, 128}};
    EXPECT_EQ(tensor2d.rank(), 2);
    EXPECT_EQ(tensor3d.rank(), 3);
    EXPECT_EQ(tensor4d.rank(), 4);
}

TEST(TestTensor, TestTensorSize) {
    constexpr int M = 8, N = 16, K = 32;

    int buffer[M][N][K];
    moke::tensor tensor{(int *)buffer, M, N, K};

    EXPECT_EQ(tensor.rank(), 3);
    EXPECT_EQ(tensor.bytes(), sizeof(buffer));
    EXPECT_EQ(tensor.size(), sizeof(buffer) / sizeof(int));

    EXPECT_EQ(tensor.shape(0), M);
    EXPECT_EQ(tensor.shape(1), N);
    EXPECT_EQ(tensor.shape(2), K);
    EXPECT_EQ(tensor.stride(0), sizeof(buffer[0]) / sizeof(int));
    EXPECT_EQ(tensor.stride(1), sizeof(buffer[0][0]) / sizeof(int));
    EXPECT_EQ(tensor.stride(2), 1);
}

TEST(TestTensor, TestTensorEmpty) {
    constexpr int M = 8, N = 16, K = 32;

    int buffer[M][N][K];
    moke::tensor tensor{(int *)buffer, M, N, K};
    EXPECT_EQ(tensor.data(), (int *)buffer);
    EXPECT_FALSE(tensor.empty());

    moke::tensor null_tensor{(int *)nullptr, M, N, K};
    EXPECT_EQ(null_tensor.data(), nullptr);
    EXPECT_TRUE(null_tensor.empty());

    moke::tensor empty_tensor{(int *)buffer, M, N, 0};
    EXPECT_NE(empty_tensor.data(), nullptr);
    EXPECT_TRUE(empty_tensor.empty());
}

TEST(TestTensor, TestTensorIndexing) {
    constexpr int M = 16, N = 16, K = 16;

    int buffer[M][N][K];
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                buffer[m][n][k] = m ^ n ^ k;
            }
        }
    }

    moke::tensor tensor{(int *)buffer, M, N, K};
    for (int m = 0; m < tensor.shape(0); m++) {
        for (int n = 0; n < tensor.shape(1); n++) {
            for (int k = 0; k < tensor.shape(2); k++) {
                ASSERT_EQ(tensor(m, n, k), m ^ n ^ k);
            }
        }
    }
}

inline auto *accept_tensor(const moke::tensor<int, 3> &tensor) { return tensor.data(); }

inline auto *accept_const_tensor(const moke::tensor<const int, 3> &tensor) { return tensor.data(); }

TEST(TestTensor, TestTensorConstConversion) {
    constexpr int M = 8, N = 16, K = 32;
    int buffer[M][N][K];
    moke::tensor tensor{(int *)buffer, M, N, K};
    EXPECT_EQ(accept_tensor(tensor), (int *)buffer);
    EXPECT_EQ(accept_const_tensor(tensor), (int *)buffer);
}
