#include <moke/arithmetic.hpp>
#include <moke/tensor.hpp>
#include <gtest/gtest.h>
#include <type_traits>

TEST(TestTensor, TestTensorConstConversion) {
    auto accept_tensor = [](const moke::tensor<int, 3> &tensor) {
        return tensor.data();
    };

    auto accept_const_tensor = [](const moke::tensor<const int, 3> &tensor) {
        return tensor.data();
    };

    auto accept_any_const_tensor = []<class T>(const moke::tensor<const T, 3> &tensor) {
        return tensor.data();
    };

    constexpr int M = 8, N = 16, K = 32;
    int buffer[M][N][K];
    moke::tensor tensor{(int *)buffer, M, N, K};
    EXPECT_EQ(accept_tensor(tensor), (int *)buffer);
    EXPECT_EQ(accept_const_tensor(tensor), (int *)buffer);
    EXPECT_EQ(accept_any_const_tensor(moke::tensor_cast<const int>(tensor)), (int *)buffer);
    EXPECT_EQ(accept_any_const_tensor(moke::tensor_cast(tensor)), (int *)buffer);
}

TEST(TestTensor, TestTensorConstruction) {
    int *buffer = nullptr;

    // construct with default layout policy (right major)
    moke::tensor t2d{buffer, 16, 32};
    moke::tensor t3d{buffer, 16, 32u, 64l};
    moke::tensor t4d{buffer, 16, 32u, 64l, 128ull};
    EXPECT_EQ(t2d.rank(), 2);
    EXPECT_EQ(t3d.rank(), 3);
    EXPECT_EQ(t4d.rank(), 4);

    // construct with right major layout policy
    moke::right_major_tensor rt2d{buffer, 16, 32};
    moke::right_major_tensor rt3d{buffer, 16, 32u, 64l};
    moke::right_major_tensor rt4d{buffer, 16, 32u, 64l, 128ull};
    EXPECT_EQ(rt2d.rank(), 2);
    EXPECT_EQ(rt3d.rank(), 3);
    EXPECT_EQ(rt4d.rank(), 4);

    // // construct with left major layout policy
    moke::left_major_tensor lt2d{buffer, 16, 32};
    moke::left_major_tensor lt3d{buffer, 16, 32u, 64l};
    moke::left_major_tensor lt4d{buffer, 16, 32u, 64l, 128ull};
    EXPECT_EQ(lt2d.rank(), 2);
    EXPECT_EQ(lt3d.rank(), 3);
    EXPECT_EQ(lt4d.rank(), 4);
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

TEST(TestTensor, TestLeftMajorTensorSize) {
    constexpr int M = 8, N = 16, K = 32;

    int buffer[M][N][K];
    moke::left_major_tensor tensor{(int *)buffer, K, N, M};

    EXPECT_EQ(tensor.rank(), 3);
    EXPECT_EQ(tensor.bytes(), sizeof(buffer));
    EXPECT_EQ(tensor.size(), sizeof(buffer) / sizeof(int));

    EXPECT_EQ(tensor.shape(0), K);
    EXPECT_EQ(tensor.shape(1), N);
    EXPECT_EQ(tensor.shape(2), M);
    EXPECT_EQ(tensor.stride(0), 1);
    EXPECT_EQ(tensor.stride(1), sizeof(buffer[0][0]) / sizeof(int));
    EXPECT_EQ(tensor.stride(2), sizeof(buffer[0]) / sizeof(int));
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
    constexpr int M = 8, N = 16, K = 32;

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

    moke::left_major_tensor l_tensor{tensor.data(), K, N, M};
    for (int m = 0; m < tensor.shape(0); m++) {
        for (int n = 0; n < tensor.shape(1); n++) {
            for (int k = 0; k < tensor.shape(2); k++) {
                ASSERT_EQ(l_tensor(k, n, m), tensor(m, n, k));
            }
        }
    }
}
