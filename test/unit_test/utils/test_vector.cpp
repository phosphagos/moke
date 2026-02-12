#include "testing_utils.hpp"
#include <moke/moke.hpp>
#include <gtest/gtest.h>

using namespace moke;
using dtypes = moke::type_tuple<float, double, half_t, bfloat16_t>;
constexpr size_t length = 128;

template <class T> class TestVector : public testing::Test {};
using TestVectorParams = test::GTestTypeList<dtypes>;
TYPED_TEST_SUITE(TestVector, TestVectorParams);

TYPED_TEST(TestVector, TestVectorAllocation) {
    host_vector<TypeParam> hvec(length);
    EXPECT_TRUE(hvec);
    EXPECT_FALSE(hvec.empty());
    EXPECT_EQ(hvec.size(), length);
    EXPECT_EQ(hvec.bytes(), length * sizeof(TypeParam));

    memory_set(host_memory, hvec.data(), 1.0, length);
    for (auto val : hvec) { EXPECT_EQ(val, TypeParam(1.0)); }
}

TYPED_TEST(TestVector, TestCopy) {
    auto v0 = host_vector<TypeParam>{length} | fill::random{};

    host_vector<TypeParam> v1{v0};
    EXPECT_NE(v0.data(), v1.data());
    for (int i = 0; i < length; i++) { ASSERT_EQ(v1[i], v0[i]); }

    memory_set(host_memory, v0.data(), 0.0, length);
    for (int i = 0; i < length; i++) { ASSERT_IN(v1[i], TypeParam(0.0), TypeParam(1.0)); }

    v1 = v0;
    EXPECT_NE(v0.data(), v1.data());
    memory_set(host_memory, v0.data(), 0.0, length);
    for (int i = 0; i < length; i++) {
        ASSERT_EQ(v1[i], v0[i]);
        ASSERT_EQ(v1[i], TypeParam(0.0));
    }
}

TYPED_TEST(TestVector, TestMove) {
    auto v0 = host_vector<TypeParam>{length} | fill::random{};

    host_vector<TypeParam> v1{std::move(v0)};
    ASSERT_FALSE(v0);
    ASSERT_TRUE(v1);
    for (int i = 0; i < length; i++) { ASSERT_IN(v1[i], TypeParam(0.0), TypeParam(1.0)); }

    v0 = std::move(v1);
    ASSERT_TRUE(v0);
    ASSERT_FALSE(v1);
    for (int i = 0; i < length; i++) { ASSERT_IN(v0[i], TypeParam(0.0), TypeParam(1.0)); }

    v0.clear();
    ASSERT_FALSE(v0);
    ASSERT_FALSE(v1);
}

TYPED_TEST(TestVector, TestHeterogenuousCopy) {
    auto hvec0 = host_vector<TypeParam>{length} | fill::random{};
    device_vector dvec{hvec0};

    ASSERT_TRUE(dvec);
    ASSERT_TRUE(hvec0);
    ASSERT_EQ(hvec0.size(), dvec.size());
    ASSERT_NE(hvec0.data(), dvec.data());

    host_vector hvec1{dvec};
    ASSERT_TRUE(hvec1);
    ASSERT_TRUE(dvec);
    ASSERT_EQ(hvec1.size(), dvec.size());
    ASSERT_NE(hvec1.data(), dvec.data());
    for (int i = 0; i < length; i++) {
        ASSERT_IN(hvec1[i], TypeParam(0.0), TypeParam(1.0));
        hvec1[i] = 0.0;
    }

    dvec = hvec1;
    ASSERT_TRUE(dvec);
    ASSERT_TRUE(hvec0);
    ASSERT_EQ(hvec0.size(), dvec.size());
    ASSERT_NE(hvec0.data(), dvec.data());

    hvec0 = dvec;
    ASSERT_TRUE(dvec);
    ASSERT_TRUE(hvec0);
    ASSERT_EQ(hvec0.size(), dvec.size());
    ASSERT_NE(hvec0.data(), dvec.data());
    for (int i = 0; i < length; i++) { ASSERT_EQ(hvec0[i], TypeParam(0.0)); }
}
