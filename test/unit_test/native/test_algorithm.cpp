#include "testing_utils.hpp"
#include <moke/moke.hpp>
#include <gtest/gtest.h>

using dtypes = moke::type_tuple<float, double, half_t, bfloat16_t>;
constexpr size_t length = 128;

template <class T> class TestAlgorithm : public testing::Test {};
using TestAlgorithmParams = moke::test::GTestProduction<memory_spaces, dtypes>;
TYPED_TEST_SUITE(TestAlgorithm, TestAlgorithmParams);

TYPED_TEST(TestAlgorithm, TestFillRandom) {
    using memory_space_t = moke::get_type<TypeParam, 0>;
    constexpr memory_space_t memory_space{};
    using dtype = moke::get_type<TypeParam, 1>;
    using vector = moke::vector<dtype, memory_space_t>;
    using host_vector = moke::host_vector<dtype>;

    vector vec0(length);
    moke::fill_random(memory_space, vec0.data(), length, 0.0, 1.0, 0);

    host_vector hvec0{vec0};
    for (int i = 0; i < length; i++) { ASSERT_IN(hvec0[i], dtype(0.0), dtype(1.0)); }
}

TYPED_TEST(TestAlgorithm, TestCompareClose) {
    using memory_space_t = moke::get_type<TypeParam, 0>;
    constexpr memory_space_t memory_space{};
    using dtype = moke::get_type<TypeParam, 1>;
    using vector = moke::vector<dtype, memory_space_t>;

    vector vec0(length), vec1(length);
    EXPECT_NE(vec0.data(), vec1.data());

    moke::memory_set(memory_space, vec0.data(), 1.0, length);
    moke::memory_set(memory_space, vec1.data(), 1.4, length);
    EXPECT_FALSE(moke::memory_compare(memory_space, vec0.data(), vec1.data(), length));
    EXPECT_TRUE(moke::compare_all_close(memory_space, vec0.data(), vec1.data(), length, std::false_type{}, 0.5));
    EXPECT_TRUE(moke::compare_all_close(memory_space, vec0.data(), vec1.data(), length, std::true_type{}, 0.5));

    moke::memory_set(memory_space, vec0.data(), 10.0, length);
    moke::memory_set(memory_space, vec1.data(), 14.0, length);
    EXPECT_FALSE(moke::memory_compare(memory_space, vec0.data(), vec1.data(), length));
    EXPECT_FALSE(moke::compare_all_close(memory_space, vec0.data(), vec1.data(), length, std::false_type{}, 0.5));
    EXPECT_TRUE(moke::compare_all_close(memory_space, vec0.data(), vec1.data(), length, std::true_type{}, 0.5));

    moke::memory_set(memory_space, vec1.data(), 4.0, length);
    EXPECT_FALSE(moke::memory_compare(memory_space, vec0.data(), vec1.data(), length));
    EXPECT_FALSE(moke::compare_all_close(memory_space, vec0.data(), vec1.data(), length, std::false_type{}, 0.5));
    EXPECT_FALSE(moke::compare_all_close(memory_space, vec0.data(), vec1.data(), length, std::true_type{}, 0.5));
}
