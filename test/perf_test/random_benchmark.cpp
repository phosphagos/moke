#include "testing_utils.hpp"
#include <gtest/gtest.h>

using namespace moke;

using DataTypes = testing::Types<float, double, moke::half_t, moke::bfloat16_t>;

template <class T> class RandomBenchmark : public testing::Test {};
TYPED_TEST_SUITE(RandomBenchmark, DataTypes);

TYPED_TEST(RandomBenchmark, TestRandomFill) {
    using dtype = TypeParam;
    for (size_t length : {64_Mi, 128_Mi, 256_Mi, 512_Mi, 1_Gi}) {
        size_t size = length * sizeof(dtype);
        printf("=== RANDOM_FILL (length=%zdM, bytes=%zdMB) ===\n", length >> 20, size >> 20);

        device_profiler profiler{};
        auto buffer = device_vector<dtype>(length);

        for (auto _ : profiler) { fill_random(device_memory, buffer.data(), length, -1.0, 1.0, 0); }
        auto time = profiler.elapsed(std::micro{});
        auto perf = profiler.perf(length, std::giga{});
        auto bw = profiler.perf(size, std::giga{});
        printf("    elapsed time: %lgus\n", time);
        printf("    compute perf: %lgGFLOPS\n", perf);
        printf("       bandwidth: %lgGB/s\n", bw);
    }
}

TYPED_TEST(RandomBenchmark, TestRandomFill2Bit) {
    using dtype = TypeParam;
    for (size_t length : {64_Mi, 128_Mi, 256_Mi, 512_Mi, 1_Gi}) {
        size_t size = length * sizeof(dtype);
        printf("=== RANDOM_FILL (length=%zdM, bytes=%zdMB) ===\n", length >> 20, size >> 20);

        device_profiler profiler{};
        auto buffer = device_vector<dtype>(length);

        for (auto _ : profiler) { fill_random(device_memory, buffer.data(), length, -1.0, 1.0, 0, 2); }
        auto time = profiler.elapsed(std::micro{});
        auto perf = profiler.perf(length, std::giga{});
        auto bw = profiler.perf(size, std::giga{});
        printf("    elapsed time: %lgus\n", time);
        printf("    compute perf: %lgGFLOPS\n", perf);
        printf("       bandwidth: %lgGB/s\n", bw);
    }
}
