#include "testing_utils.hpp"
#include <gtest/gtest.h>

using namespace moke;
using dtypes = moke::type_tuple<float, double, half_t, bfloat16_t>;
constexpr size_t length = 1 << 20;

template <class T> class TestFunctional : public testing::Test {};
using TestFunctionalParams = moke::test::GTestProduction<memory_spaces, dtypes>;
TYPED_TEST_SUITE(TestFunctional, TestFunctionalParams);

TYPED_TEST(TestFunctional, TestFillConstant) {
    using memory_space_t = moke::get_type<TypeParam, 0>;
    using dtype = moke::get_type<TypeParam, 1>;
    using vector = moke::vector<dtype, memory_space_t>;
    using host_vector = moke::host_vector<dtype>;

    for (dtype value : {0.0f, 1.0f, 1.5f}) {
        vector vec = vector(length) | fill::constant{value};
        host_vector hvec = vec;
        for (int i = 0; i < length; i++) { ASSERT_EQ(hvec[i], value); }
    }
}

TYPED_TEST(TestFunctional, TestFillRandom) {
    using memory_space_t = moke::get_type<TypeParam, 0>;
    using dtype = moke::get_type<TypeParam, 1>;
    using vector = moke::vector<dtype, memory_space_t>;
    using host_vector = moke::host_vector<dtype>;

    auto vec0 = vector(length) | fill::random();
    auto vec1 = vector(length) | fill::random(128);
    auto vec2 = vector(length) | fill::random{-1, 1};
    host_vector hvec0{vec0}, hvec1{vec1}, hvec2{vec2};

    for (int i = 0; i < length; i++) { ASSERT_IN(hvec0[i], dtype(0.0), dtype(1.0)); }
    for (int i = 0; i < length; i++) { ASSERT_IN(hvec1[i], dtype(0.0), dtype(128.0)); }
    for (int i = 0; i < length; i++) { ASSERT_IN(hvec2[i], dtype(-1.0), dtype(1.0)); }
}

TYPED_TEST(TestFunctional, TestFillRandomWithBits) {
    using memory_space_t = moke::get_type<TypeParam, 0>;
    using dtype = moke::get_type<TypeParam, 1>;
    using vector = moke::vector<dtype, memory_space_t>;
    using host_vector = moke::host_vector<dtype>;

    auto vec0 = vector(length) | fill::random<2>();
    auto vec1 = vector(length) | fill::random<2>(128);
    auto vec2 = vector(length) | fill::random<2>{-1, 1};
    auto vec3 = vector(length) | fill::random<2>{-128, 128};
    host_vector hvec0{vec0}, hvec1{vec1}, hvec2{vec2}, hvec3{vec3};

    for (int i = 0; i < length; i++) { ASSERT_IN(hvec0[i], dtype(0.0), dtype(1.0)); }
    for (int i = 0; i < length; i++) { ASSERT_IN(hvec1[i], dtype(0.0), dtype(128.0)); }
    for (int i = 0; i < length; i++) { ASSERT_IN(hvec2[i], dtype(-1.0), dtype(1.0)); }
    for (int i = 0; i < length; i++) { ASSERT_IN(hvec3[i], dtype(-128.0), dtype(128.0)); }
}

TYPED_TEST(TestFunctional, TestCompare) {
    using memory_space_t = moke::get_type<TypeParam, 0>;
    constexpr memory_space_t memory_space{};
    using dtype = moke::get_type<TypeParam, 1>;
    using vector = moke::vector<dtype, memory_space_t>;

    vector vec0(length), vec1(length);
    EXPECT_NE(vec0.data(), vec1.data());

    vec0 | fill::constant(0.0);
    vec1 | fill::constant(0.0);
    EXPECT_TRUE(vec1 | compare::all_equals(vec0));
    EXPECT_TRUE(vec1 | compare::all_close(vec0, 0.5));

    vec0 | fill::constant(10.0);
    vec1 | fill::constant(14.0);
    EXPECT_FALSE(vec1 | compare::all_equals(vec0));
    EXPECT_FALSE(vec1 | compare::all_close(vec0, 0.5, std::false_type{}));
    EXPECT_TRUE(vec1 | compare::all_close(vec0, 0.5, std::true_type{}));

    vec1 | fill::constant(4.0);
    EXPECT_FALSE(vec1 | compare::all_equals(vec0));
    EXPECT_FALSE(vec1 | compare::all_close(vec0, 0.5, std::false_type{}));
    EXPECT_FALSE(vec1 | compare::all_close(vec0, 0.5, std::true_type{}));
}
