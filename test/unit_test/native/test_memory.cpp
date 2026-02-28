#include "testing_utils.hpp"
#include <gtest/gtest.h>

using namespace moke;
using dtypes = moke::type_tuple<float, double, half_t, bfloat16_t>;
constexpr size_t length = 128 << 20;

template <class T> class TestMemory : public testing::Test {};
using TestMemoryParams = test::GTestProduction<memory_spaces, dtypes>;
TYPED_TEST_SUITE(TestMemory, TestMemoryParams);

TYPED_TEST(TestMemory, TestMalloc) {
    using memory_space_t = get_type<TypeParam, 0>;
    using dtype = get_type<TypeParam, 1>;

    auto buffer = memory_alloc<dtype>(memory_space_t{}, length);
    EXPECT_TRUE(buffer);
    memory_free(memory_space_t{}, buffer);
}

TYPED_TEST(TestMemory, TestMemorySet) {
    using memory_space_t = get_type<TypeParam, 0>;
    using dtype = get_type<TypeParam, 1>;

    constexpr memory_space_t memory_space{};
    constexpr float value{1};

    auto buffer = memory_alloc<dtype>(memory_space, length);
    auto host_buffer = host_malloc<dtype>(length);

    memory_set(memory_space, buffer, value, length);
    memory_copy(host_memory, memory_space, host_buffer, buffer, length);
    for (int i = 0; i < length; i++) { ASSERT_EQ(host_buffer[i], dtype(value)); }

    host_free(host_buffer);
    memory_free(memory_space, buffer);
}

TYPED_TEST(TestMemory, TestMemoryCompare) {
    using memory_space_t = moke::get_type<TypeParam, 0>;
    using dtype = moke::get_type<TypeParam, 1>;

    constexpr memory_space_t memory_space{};
    auto buf0 = memory_alloc<dtype>(memory_space, length);
    auto buf1 = memory_alloc<dtype>(memory_space, length);
    auto hbuf = host_malloc<dtype>(length);

    for (float value : {0.0, 0.4, 1.0}) {
        EXPECT_NE(buf0, buf1);

        moke::memory_set(memory_space, buf0, value, length);
        moke::memory_set(memory_space, buf1, value, length);
        EXPECT_TRUE(moke::memory_compare(memory_space, buf0, buf1, length));

        moke::memory_set(memory_space, buf1, 1 - value, length);
        EXPECT_FALSE(moke::memory_compare(memory_space, buf0, buf1, length));
    }

    host_free(hbuf);
    memory_free(memory_space, buf1);
    memory_free(memory_space, buf0);
}

template <class T> class TestMemoryCopy : public testing::Test {};
using TestMemoryCopyParams = test::GTestProduction<memory_spaces, memory_spaces, dtypes>;
TYPED_TEST_SUITE(TestMemoryCopy, TestMemoryCopyParams);

TYPED_TEST(TestMemoryCopy, TestMemoryCopy) {
    using src_memory_t = get_type<TypeParam, 0>;
    using dest_memory_t = get_type<TypeParam, 1>;
    using dtype = get_type<TypeParam, 2>;

    constexpr src_memory_t src_memory{};
    constexpr dest_memory_t dest_memory{};
    constexpr float value{12};

    auto src_buf = memory_alloc<dtype>(src_memory, length);
    auto dest_buf = memory_alloc<dtype>(dest_memory, length);
    auto host_buf = host_malloc<dtype>(length);

    memory_set(src_memory, src_buf, value, length);
    memory_copy(dest_memory, src_memory, dest_buf, src_buf, length);
    memory_copy(host_memory, dest_memory, host_buf, dest_buf, length);
    for (int i = 0; i < length; i++) { ASSERT_EQ(host_buf[i], dtype(value)); }

    host_free(host_buf);
    memory_free(dest_memory, dest_buf);
    memory_free(src_memory, src_buf);
}
