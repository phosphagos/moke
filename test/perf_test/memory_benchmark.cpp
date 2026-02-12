#include "testing_utils.hpp"
#include <gtest/gtest.h>
#include <moke/moke.hpp>

using namespace moke;

using MemsetValues = type_tuple<
        C<(uint8_t)0xEE>, C<(uint16_t)0x80FF>,
        C<(uint32_t)0x66CCFF>, C<(uint64_t)0x0066CCFF'00EE0000>>;
using TestMemoryParams = test::GTestTypeList<MemsetValues>;

template <class T> class MemoryBenchmark : public testing::Test {};
TYPED_TEST_SUITE(MemoryBenchmark, TestMemoryParams);

TYPED_TEST(MemoryBenchmark, TestMemorySet) {
    using dtype = TypeParam::value_type;
    constexpr auto value = TypeParam::value;

    for (size_t size : {128_Mi, 256_Mi, 512_Mi, 1_Gi, 2_Gi}) {
        size_t length = size / sizeof(dtype);
        printf("=== MEMORY_SET (length=%zdM, bytes=%zdMB) ===\n", length >> 20, size >> 20);

        device_profiler profiler{};
        auto buffer = device_vector<dtype>(length);

        for (auto _ : profiler) { memory_set(device_memory, buffer.data(), value, length); }
        auto time = profiler.elapsed(std::micro{});
        auto bw = profiler.perf(size, std::giga{});
        printf("    elapsed time: %lgus\n", time);
        printf("       bandwidth: %lgGB/s\n", bw);
    }
}

TYPED_TEST(MemoryBenchmark, TestMemoryCopy) {
    using dtype = TypeParam::value_type;
    constexpr auto value = TypeParam::value;

    for (size_t size : {64_Mi, 128_Mi, 256_Mi, 1_Gi}) {
        size_t length = size / sizeof(dtype);
        printf("=== MEMORY_COPY (length=%zdM, bytes=%zdMB) ===\n", length >> 20, size >> 20);

        device_profiler profiler{};
        auto buf0 = device_vector<dtype>(length) | fill::constant<dtype>(value);
        auto buf1 = device_vector<dtype>(length);

        for (auto _ : profiler) { memory_copy(device_memory, buf1.data(), buf0.data(), length); }
        auto time = profiler.elapsed(std::micro{});
        auto bw = profiler.perf(size * 2, std::giga{});
        printf("    elapsed time: %lgus\n", time);
        printf("       bandwidth: %lgGB/s\n", bw);
    }
}