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
    using Coord = decltype(coord);

    // size of the tuple
    static_assert(Coord::size == 3);
    static_assert(constant_tuple<>::size == 0);

    // indexing tuple statically, using constant type
    EXPECT_EQ(moke::get_value(coord, C<0>{}), 16);
    EXPECT_EQ(moke::get_value(coord, C<1>{}), 32);
    EXPECT_EQ(moke::get_value(coord, C<2>{}), 64);

    // indexing tuple statically, using template parameter
    EXPECT_EQ((moke::get_value<Coord, 0>()), 16);
    EXPECT_EQ((moke::get_value<Coord, 1>()), 32);
    EXPECT_EQ((moke::get_value<Coord, 2>()), 64);

    // get_t on a constant_tuple yields the constant type
    static_assert(std::same_as<get_t<Coord, 0>, C<16>>);
    static_assert(std::same_as<get_t<Coord, 1>, C<32>>);
    static_assert(std::same_as<get_t<Coord, 2>, C<64>>);
}

TEST(TestMeta, TestTypeTuple) {
    using tuple = type_tuple<float, int, char>;
    static_assert(tuple::size == 3);
    static_assert(type_tuple<>::size == 0);

    static_assert(std::same_as<get_t<tuple, 0>, float>);
    static_assert(std::same_as<get_t<tuple, 1>, int>);
    static_assert(std::same_as<get_t<tuple, 2>, char>);
}

TEST(TestMeta, TestConcat) {
    static_assert(std::same_as<
                  concat_t<constant_tuple<1, 2>, constant_tuple<3, 4>>,
                  constant_tuple<1, 2, 3, 4>>);
    static_assert(std::same_as<
                  concat_t<constant_tuple<1, 2>, C<3>>,
                  constant_tuple<1, 2, 3>>);
    static_assert(std::same_as<
                  concat_t<constant_tuple<>, constant_tuple<5>>,
                  constant_tuple<5>>);
    static_assert(std::same_as<
                  concat_t<constant_tuple<>, C<5>>,
                  constant_tuple<5>>);
}

TEST(TestMeta, TestShift) {
    static_assert(std::same_as<shift_t<constant_tuple<10, 20, 30>>, constant_tuple<20, 30>>);
    static_assert(std::same_as<shift_t<constant_tuple<42>>, constant_tuple<>>);
    static_assert(std::same_as<shift_t<shift_t<constant_tuple<1, 2, 3>>>, constant_tuple<3>>);
}

TEST(TestMeta, TestSwap) {
    using list = constant_tuple<10, 20, 30, 40>;
    using tuple = constant_tuple<true, 5, 'a'>;

    static_assert(std::same_as<swap_t<list, 0, 3>, constant_tuple<40, 20, 30, 10>>);
    static_assert(std::same_as<swap_t<list, 1, 2>, constant_tuple<10, 30, 20, 40>>);
    static_assert(std::same_as<swap_t<list, 2, 3>, constant_tuple<10, 20, 40, 30>>);

    static_assert(std::same_as<swap_t<tuple, 0, 1>, constant_tuple<5, true, 'a'>>);
    static_assert(std::same_as<swap_t<tuple, 0, 2>, constant_tuple<'a', 5, true>>);
    static_assert(std::same_as<swap_t<tuple, 1, 2>, constant_tuple<true, 'a', 5>>);

    static_assert(std::same_as<swap_t<list, 1, 1>, list>);
    static_assert(std::same_as<swap_t<list, 0, 3>, swap_t<list, 3, 0>>);
    static_assert(std::same_as<swap_t<swap_t<list, 0, 2>, 0, 2>, list>);
    static_assert(std::same_as<swap_t<constant_tuple<7>, 0, 0>, constant_tuple<7>>);
}
