#include <moke/meta.hpp>
#include <gtest/gtest.h>

using namespace moke;
using namespace moke::literals;

inline bool test_constant(moke::true_t) { return true; }
inline bool test_constant(moke::false_t) { return false; }

TEST(TestMeta, TestConstant) {
    constexpr int v0 = C<128>{};
    auto v1 = C<256>{};

    static_assert(v0 == 128);
    static_assert(v1 == 256);

    constexpr auto b0 = TRUE;
    constexpr auto b1 = FALSE;
    static_assert(std::same_as<decltype(b0), const true_t>);
    static_assert(std::same_as<decltype(b1), const false_t>);
    static_assert(b0 == true);
    static_assert(b1 == false);
    EXPECT_TRUE(test_constant(TRUE));
    EXPECT_FALSE(test_constant(FALSE));

    static_assert(128_ic == C<128>{});
    static_assert(1'024_ic == C<1024>{});
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
}

TEST(TestMeta, TestTypeTuple) {
    using tuple = type_tuple<float, int, char>;
    static_assert(std::same_as<get_t<tuple, 0>, float>);
    static_assert(std::same_as<get_t<tuple, 1>, int>);
    static_assert(std::same_as<get_t<tuple, 2>, char>);
}
