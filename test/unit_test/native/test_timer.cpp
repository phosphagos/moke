#include "testing_utils.hpp"
#include <gtest/gtest.h>

#if defined __clang__
#define OPTNONE [[clang::optnone]]
#elif defined __GNUC__
#define OPTNONE [[gnu::optimize("O0")]]
#else
#define OPTNONE
#endif

using namespace moke;
TEST(TestTimer, TestDurationCast) {
    duration<> s{1};
    auto ms = duration_cast<std::milli>(s);
    EXPECT_EQ(ms.count(), 1e3);

    auto ks = duration_cast<std::kilo>(s);
    EXPECT_EQ(ks.count(), 1e-3);
}

TEST(TestTimer, TestHostTimer) {
    constexpr size_t length = 1 << 20;
    auto vec0 = host_vector<float>(length) | fill::random{};
    auto vec1 = host_vector<float>(length);

    auto time = host_timer::invoke<std::micro>([&] OPTNONE () { vec1 = vec0; });
    printf("host_timer: copies %zd fp32 in %.0lfus\n", length, time.count());
    printf("bandwidth: %lgGB/s\n", length * 2 * sizeof(float) / time.count() / 1000);
    EXPECT_GT(time.count(), 0);
}

TEST(TestTimer, TestDeviceTimer) {
    constexpr size_t length = 1 << 20;
    auto vec0 = device_vector<float>(length) | fill::random{};
    auto vec1 = device_vector<float>(length);
    sync_device();

    auto time = device_timer::invoke<std::micro>([&] { vec1 = vec0; });
    printf("device_timer: copies %zd fp32 in %lgus\n", length, time.count());
    printf("bandwidth: %lgGB/s\n", length * 2 * sizeof(float) / time.count() / 1000);
    EXPECT_GT(time.count(), 0);
}
