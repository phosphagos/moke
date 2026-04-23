#include <moke/tensor.hpp>
#include <gtest/gtest.h>


TEST(TestLayout, TestCtor) {
    moke::layout layout_1(1, 16, 32);
    EXPECT_EQ(layout_1.rank(), 3);
    EXPECT_EQ(layout_1.shape(0), 1);
    EXPECT_EQ(layout_1.shape(1), 16);
    EXPECT_EQ(layout_1.shape(2), 32);
    EXPECT_EQ(layout_1.stride(0), 16 * 32);
    EXPECT_EQ(layout_1.stride(1), 32);
    EXPECT_EQ(layout_1.stride(2), 1);

    moke::layout layout_2({1, 32, 16});
    EXPECT_EQ(layout_2.rank(), 3);
    EXPECT_EQ(layout_2.shape(0), 1);
    EXPECT_EQ(layout_2.shape(1), 32);
    EXPECT_EQ(layout_2.shape(2), 16);
    EXPECT_EQ(layout_2.stride(0), 16 * 32);
    EXPECT_EQ(layout_2.stride(1), 16);
    EXPECT_EQ(layout_2.stride(2), 1);
}

TEST(TestLayout, TestIndex2Offset) {
    moke::layout layout(16, 32, 64);
    EXPECT_EQ(layout(1), 32 * 64);
    EXPECT_EQ(layout(0, 1), 64);
    EXPECT_EQ(layout(0, 0, 1), 1);

    EXPECT_EQ(layout(4, 2, 3), (4 * 32 + 2) * 64 + 3);
    EXPECT_EQ(layout(8, 6, 4), 4 + 6 * 64 + 8 * 32 * 64);
}

TEST(TestLayout, TestCopyReset) {
    moke::layout<3> layout{};
    EXPECT_EQ(layout.size(), 0);
    EXPECT_TRUE(layout.empty());

    layout = {16, 8, 4};
    EXPECT_EQ(layout.size(), 16 * 8 * 4);
    EXPECT_FALSE(layout.empty());
}