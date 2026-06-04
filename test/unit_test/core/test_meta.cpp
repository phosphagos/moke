#include <moke/meta.hpp>
#include <gtest/gtest.h>

using namespace moke;

inline bool test_constant(moke::true_t) { return true; }
inline bool test_constant(moke::false_t) { return false; }

TEST(TestMeta, TestConstant) {
    constexpr int v0 = C<128>{};
    static_assert(v0 == 128);

    auto v1 = C<256>{};
    static_assert(v1 == 256);

    constexpr auto b0 = TRUE;
    constexpr auto b1 = FALSE;
    static_assert(b0 == true);
    static_assert(b1 == false);

    EXPECT_TRUE(test_constant(TRUE));
    EXPECT_FALSE(test_constant(FALSE));
}

TEST(TestMeta, TestConstantTuple) {
    constant_tuple<16, 32, 64> coord{};

    // indexing tuple statically, using constant type
    EXPECT_EQ(moke::get_value(coord, C<0>{}), 16);
    EXPECT_EQ(moke::get_value(coord, C<1>{}), 32);
    EXPECT_EQ(moke::get_value(coord, C<2>{}), 64);

    // indexing tuple statically, using template parameter
    using Coord = decltype(coord);
    EXPECT_EQ((moke::get_value<Coord, 0>()), 16);
    EXPECT_EQ((moke::get_value<Coord, 1>()), 32);
    EXPECT_EQ((moke::get_value<Coord, 2>()), 64);

    // indexing tuple dynamically
    for (int i = 0; i < 3; i++) { EXPECT_EQ(get_value(coord, i), 1 << (i + 4)); }

    // returns 0 if index overflow
    for (int i = 3; i < 10; i++) { EXPECT_EQ(get_value(coord, i), 0); }
}

TEST(TestMeta, TestTypeTuple) {
    using tuple = type_tuple<float, int, char>;
    static_assert(std::same_as<get_t<tuple, 0>, float>);
    static_assert(std::same_as<get_t<tuple, 1>, int>);
    static_assert(std::same_as<get_t<tuple, 2>, char>);
}

TEST(TestMeta, TestTypeTupleProduct) {
    // 2x2 product
    using tuple_00 = type_tuple<float, int>;
    using tuple_01 = type_tuple<char, double>;
    using prod_0 = product_t<tuple_00, tuple_01>;
    static_assert(std::same_as<get_t<prod_0, 0>, type_tuple<float, char>>);
    static_assert(std::same_as<get_t<prod_0, 1>, type_tuple<float, double>>);
    static_assert(std::same_as<get_t<prod_0, 2>, type_tuple<int, char>>);
    static_assert(std::same_as<get_t<prod_0, 3>, type_tuple<int, double>>);

    // 3x3 product
    using tuple_10 = type_tuple<uint16_t, uint32_t, uint64_t>;
    using tuple_11 = type_tuple<int16_t, int32_t, int64_t>;
    using prod_1 = product_t<tuple_10, tuple_11>;
    static_assert(std::same_as<get_t<prod_1, 0>, type_tuple<uint16_t, int16_t>>);
    static_assert(std::same_as<get_t<prod_1, 1>, type_tuple<uint16_t, int32_t>>);
    static_assert(std::same_as<get_t<prod_1, 2>, type_tuple<uint16_t, int64_t>>);
    static_assert(std::same_as<get_t<prod_1, 3>, type_tuple<uint32_t, int16_t>>);
    static_assert(std::same_as<get_t<prod_1, 4>, type_tuple<uint32_t, int32_t>>);
    static_assert(std::same_as<get_t<prod_1, 5>, type_tuple<uint32_t, int64_t>>);
    static_assert(std::same_as<get_t<prod_1, 6>, type_tuple<uint64_t, int16_t>>);
    static_assert(std::same_as<get_t<prod_1, 7>, type_tuple<uint64_t, int32_t>>);
    static_assert(std::same_as<get_t<prod_1, 8>, type_tuple<uint64_t, int64_t>>);

    // 2x2x2 product
    using tuple_20 = type_tuple<float, double>;
    using tuple_21 = type_tuple<int32_t, int64_t>;
    using tuple_22 = type_tuple<uint32_t, uint64_t>;
    using prod_2 = product_t<tuple_20, tuple_21, tuple_22>;
    static_assert(std::same_as<get_t<prod_2, 0>, type_tuple<float, int32_t, uint32_t>>);
    static_assert(std::same_as<get_t<prod_2, 1>, type_tuple<float, int32_t, uint64_t>>);
    static_assert(std::same_as<get_t<prod_2, 2>, type_tuple<float, int64_t, uint32_t>>);
    static_assert(std::same_as<get_t<prod_2, 3>, type_tuple<float, int64_t, uint64_t>>);
    static_assert(std::same_as<get_t<prod_2, 4>, type_tuple<double, int32_t, uint32_t>>);
    static_assert(std::same_as<get_t<prod_2, 5>, type_tuple<double, int32_t, uint64_t>>);
    static_assert(std::same_as<get_t<prod_2, 6>, type_tuple<double, int64_t, uint32_t>>);
    static_assert(std::same_as<get_t<prod_2, 7>, type_tuple<double, int64_t, uint64_t>>);
}
