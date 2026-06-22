#include <moke/utils.hpp>
#include <gtest/gtest.h>

using namespace moke;

TEST(TestMetaUtils, TestTypeTupleConcat) {
    static_assert(std::same_as<
                  concat_t<type_tuple<float, int>, type_tuple<char, double>>,
                  type_tuple<float, int, char, double>>);
    static_assert(std::same_as<
                  concat_t<type_tuple<float, int>, char>,
                  type_tuple<float, int, char>>);
    static_assert(std::same_as<
                  concat_t<type_tuple<>, type_tuple<int>>,
                  type_tuple<int>>);
    static_assert(std::same_as<
                  concat_t<type_tuple<>, int>,
                  type_tuple<int>>);
}

TEST(TestMetaUtils, TestTypeTupleProduct) {
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
