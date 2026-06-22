#include <moke/layout/static_layout.hpp>
#include <gtest/gtest.h>

using namespace moke;
using namespace moke::literals;

TEST(TestStaticLayout, TestRightMajorLayout) {
    // (16, 16, 8) -> strides (128, 8, 1), size 2048, dim_cons rank - 1
    using layout = decltype(make_right_major_layout(constant_tuple<16, 16, 8>{}));

    static_assert(std::same_as<layout::shape, constant_tuple<16, 16, 8>>);
    static_assert(std::same_as<layout::stride, constant_tuple<16 * 8, 8, 1>>);
    static_assert(layout::rank == 3);
    static_assert(layout::size == 2048);
    static_assert(layout::dim_cons == 2);
    static_assert(layout::empty == false);

    // the constant<...> overload produces the same type
    static_assert(std::same_as<layout, decltype(make_right_major_layout(16_ic, 16_ic, 8_ic))>);
}

TEST(TestStaticLayout, TestLeftMajorLayout) {
    // (16, 16, 8) -> strides (1, 16, 256), size 2048, dim_cons 0
    using layout = decltype(make_left_major_layout(constant_tuple<16, 16, 8>{}));

    static_assert(std::same_as<layout::shape, constant_tuple<16, 16, 8>>);
    static_assert(std::same_as<layout::stride, constant_tuple<1, 16, 256>>);
    static_assert(layout::rank == 3);
    static_assert(layout::size == 2048);
    static_assert(layout::dim_cons == 0);
    static_assert(layout::empty == false);

    static_assert(std::same_as<layout, decltype(make_left_major_layout(16_ic, 16_ic, 8_ic))>);
}

TEST(TestStaticLayout, TestMakeLayout) {
    // make_layout defaults to right major
    static_assert(std::same_as<
                  decltype(make_layout(constant_tuple<16, 16, 8>{})),
                  decltype(make_right_major_layout(constant_tuple<16, 16, 8>{}))>);
    static_assert(std::same_as<
                  decltype(make_layout(16_ic, 16_ic, 8_ic)),
                  decltype(make_right_major_layout(16_ic, 16_ic, 8_ic))>);
}

TEST(TestStaticLayout, TestRightMajorOffset) {
    // (128, 8, 1): offset(a, b, c) = 128a + 8b + c
    constexpr auto layout = make_right_major_layout(constant_tuple<16, 16, 8>{});

    static_assert(layout(0, 0, 0) == 0);
    static_assert(layout(1, 0, 0) == 128);
    static_assert(layout(0, 1, 0) == 8);
    static_assert(layout(0, 0, 1) == 1);
    static_assert(layout(1, 2, 3) == 128 + 16 + 3);

    EXPECT_EQ(layout(2, 5, 7), 2 * 128 + 5 * 8 + 7);
    EXPECT_EQ(layout(15, 15, 7), 2047);
}

TEST(TestStaticLayout, TestLeftMajorOffset) {
    // (1, 16, 256): offset(a, b, c) = a + 16b + 256c
    constexpr auto layout = make_left_major_layout(constant_tuple<16, 16, 8>{});

    static_assert(layout(0, 0, 0) == 0);
    static_assert(layout(1, 0, 0) == 1);
    static_assert(layout(0, 1, 0) == 16);
    static_assert(layout(0, 0, 1) == 256);
    static_assert(layout(1, 2, 3) == 1 + 32 + 768);

    EXPECT_EQ(layout(2, 5, 7), 2 + 5 * 16 + 7 * 256);
    EXPECT_EQ(layout(15, 15, 7), 2047);
}

TEST(TestStaticLayout, TestPartialIndexing) {
    // fewer indices than rank: missing trailing coordinates are treated as 0
    constexpr auto layout = make_right_major_layout(constant_tuple<16, 16, 8>{});

    static_assert(layout() == 0);
    static_assert(layout(1) == 128);
    static_assert(layout(1, 1) == 128 + 8);
    static_assert(layout(1, 1, 1) == 128 + 8 + 1);

    EXPECT_EQ(layout(3), 3 * 128);
    EXPECT_EQ(layout(3, 4), 3 * 128 + 4 * 8);
}

TEST(TestStaticLayout, TestEdgeCases) {
    // 1-D layout
    using vec = decltype(make_right_major_layout(constant_tuple<7>{}));
    static_assert(std::same_as<vec::shape, constant_tuple<7>>);
    static_assert(std::same_as<vec::stride, constant_tuple<1>>);
    static_assert(vec::rank == 1);
    static_assert(vec::size == 7);
    static_assert(vec::dim_cons == 0);
    static_assert(vec{}(5) == 5);

    // a zero extent makes the layout empty
    static_assert(make_right_major_layout(constant_tuple<0, 4>{}).empty == true);
    static_assert(make_left_major_layout(constant_tuple<4, 0>{}).empty == true);
}
