#include <moke/static_tensor.hpp>
#include <moke/tensor.hpp>
#include <gtest/gtest.h>

using namespace moke;
using namespace moke::literals;

TEST(TestStaticTensor, TestMakerTypes) {
    // right major (16, 16, 8) -> strides (128, 8, 1)
    using rm_ten = decltype(make_right_major_tensor((float *)nullptr, constant_tuple<16, 16, 8>{}));
    static_assert(std::same_as<rm_ten::value_t, float>);
    static_assert(std::same_as<rm_ten::shape_t, constant_tuple<16, 16, 8>>);
    static_assert(std::same_as<rm_ten::stride_t, constant_tuple<128, 8, 1>>);
    static_assert(rm_ten::rank() == 3);
    static_assert(rm_ten::size() == 2048);

    // left major (16, 16, 8) -> strides (1, 16, 256)
    using lm_ten = decltype(make_left_major_tensor((float *)nullptr, constant_tuple<16, 16, 8>{}));
    static_assert(std::same_as<lm_ten::value_t, float>);
    static_assert(std::same_as<lm_ten::shape_t, constant_tuple<16, 16, 8>>);
    static_assert(std::same_as<lm_ten::stride_t, constant_tuple<1, 16, 256>>);
    static_assert(lm_ten::rank() == 3);
    static_assert(lm_ten::size() == 2048);

    // make_tensor defaults to right major
    using ten = decltype(make_tensor((float *)nullptr, constant_tuple<16, 16, 8>{}));
    static_assert(std::same_as<ten, rm_ten>);

    // the constant<N>... overloads produce the same types as the constant_tuple ones
    using rm_ten_ic = decltype(make_right_major_tensor((float *)nullptr, 16_ic, 16_ic, 8_ic));
    using lm_ten_ic = decltype(make_left_major_tensor((float *)nullptr, 16_ic, 16_ic, 8_ic));
    using ten_ic = decltype(make_tensor((float *)nullptr, 16_ic, 16_ic, 8_ic));
    static_assert(std::same_as<rm_ten_ic, rm_ten>);
    static_assert(std::same_as<lm_ten_ic, lm_ten>);
    static_assert(std::same_as<ten_ic, ten>);
}

TEST(TestStaticTensor, TestMetadata) {
    float buf[2048];
    auto t = make_right_major_tensor(buf, constant_tuple<16, 16, 8>{});
    using tensor_t = decltype(t);

    static_assert(tensor_t::RANK == 3);
    static_assert(std::same_as<tensor_t::value_t, float>);
    static_assert(t.rank() == 3);
    static_assert(t.size() == 2048);
    static_assert(t.bytes() == 2048 * sizeof(float));
}

TEST(TestStaticTensor, TestShapeAndStride) {
    float buf[2048];
    auto rm = make_right_major_tensor(buf, constant_tuple<16, 16, 8>{});
    auto lm = make_left_major_tensor(buf, constant_tuple<16, 16, 8>{});

    // dimension is a compile-time C<Dim>, queried either positionally or as a template arg
    static_assert(rm.shape(C<0>{}) == 16);
    static_assert(rm.shape(C<2>{}) == 8);
    static_assert(rm.shape<1>() == 16);

    static_assert(rm.stride(C<0>{}) == 128);
    static_assert(rm.stride(C<1>{}) == 8);
    static_assert(rm.stride(C<2>{}) == 1);

    static_assert(lm.stride<0>() == 1);
    static_assert(lm.stride<1>() == 16);
    static_assert(lm.stride<2>() == 256);
}

TEST(TestStaticTensor, TestEmptyAndData) {
    float buf[8];
    auto t = make_right_major_tensor(buf, constant_tuple<2, 4>{});
    EXPECT_EQ(t.data(), buf);
    EXPECT_FALSE(t.empty());

    // a null data pointer makes the tensor empty
    decltype(t) null_tensor{};
    EXPECT_EQ(null_tensor.data(), nullptr);
    EXPECT_TRUE(null_tensor.empty());
}

TEST(TestStaticTensor, TestFullIndexing) {
    float buf[2048] = {};

    // right major (128, 8, 1): &elem(a,b,c) == buf + 128a + 8b + c
    auto rm = make_right_major_tensor(buf, constant_tuple<16, 16, 8>{});
    rm(1, 2, 3) = 7.0f;
    EXPECT_EQ(buf[1 * 128 + 2 * 8 + 3], 7.0f);
    EXPECT_EQ(&rm(1, 2, 3), buf + 1 * 128 + 2 * 8 + 3);

    // left major (1, 16, 256): &elem(a,b,c) == buf + a + 16b + 256c
    auto lm = make_left_major_tensor(buf, constant_tuple<16, 16, 8>{});
    lm(1, 2, 3) = 9.0f;
    EXPECT_EQ(buf[1 + 2 * 16 + 3 * 256], 9.0f);
    EXPECT_EQ(&lm(1, 2, 3), buf + 1 + 2 * 16 + 3 * 256);
}

TEST(TestStaticTensor, TestPartialIndexing) {
    float buf[2048] = {};
    auto rm = make_right_major_tensor(buf, constant_tuple<16, 16, 8>{});

    // fewer indices than rank -> pointer to the start of the addressed sub-tensor
    EXPECT_EQ(rm(), buf);
    EXPECT_EQ(rm(1), buf + 128);
    EXPECT_EQ(rm(1, 2), buf + 128 + 16);
}

TEST(TestStaticTensor, TestConstConversion) {
    float buf[8] = {};
    auto t = make_right_major_tensor(buf, constant_tuple<2, 4>{});

    // mutable -> const conversion preserves the layout type and data pointer
    static_tensor<const float, decltype(t)::layout_t> ct = t;
    EXPECT_EQ(ct.data(), buf);
    static_assert(std::same_as<decltype(ct)::value_t, const float>);
}

TEST(TestStaticTensor, TestCoexistWithDynamicMakers) {
    float buf[2048] = {};

    // runtime integer shapes still select the dynamic basic_tensor makers
    auto dyn = make_tensor(buf, 16, 16, 8);
    static_assert(std::same_as<decltype(dyn), basic_tensor<float, layout::right_major<3>>>);
    EXPECT_EQ(dyn.size(), 2048u);

    // compile-time shapes select the static_tensor makers
    auto stat = make_tensor(buf, constant_tuple<16, 16, 8>{});
    static_assert(std::same_as<decltype(stat)::layout_t, static_layout<constant_tuple<16, 16, 8>, constant_tuple<128, 8, 1>, 2048, 2>>);
}
