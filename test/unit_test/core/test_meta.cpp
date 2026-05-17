#include <moke/meta.hpp>
#include <gtest/gtest.h>

using namespace moke;

constexpr bool test_constant(moke::true_t) { return true; }
constexpr bool test_constant(moke::false_t) { return false; }

TEST(TestMeta, TestConstant) {
    constexpr int v0 = C<128>{};
    static_assert(v0 == 128);

    auto v1 = C<256>{};
    static_assert(v1 == 256);

    constexpr auto b0 = TRUE;
    constexpr auto b1 = FALSE;
    static_assert(b0 == true);
    static_assert(b1 == false);

    static_assert(test_constant(TRUE));
    static_assert(test_constant(FALSE) == false);
}

TEST(TestMeta, TestConstantTuple) {
    moke::constant_tuple<16, 32, 64> coord{};

    // indexing tuple statically, using constant type
    static_assert(moke::get_value(coord, C<0>{}) == 16);
    static_assert(moke::get_value(coord, C<1>{}) == 32);
    static_assert(moke::get_value(coord, C<2>{}) == 64);

    // indexing tuple statically, using template parameter
    using Coord = decltype(coord);
    static_assert(moke::get_value<Coord, 0>() == 16);
    static_assert(moke::get_value<Coord, 1>() == 32);
    static_assert(moke::get_value<Coord, 2>() == 64);

    // indexing tuple dynamically
    for (int i = 0; i < 3; i++) { EXPECT_EQ(moke::get_value(coord, i), 1 << (i + 4)); }

    // returns 0 if index overflow
    for (int i = 3; i < 10; i++) { EXPECT_EQ(moke::get_value(coord, i), 0); }
}
