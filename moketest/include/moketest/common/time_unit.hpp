#pragma once
#include "moke/common.hpp"
#include <cstdint>

namespace moke {
enum TimeUnit {
    TIME_SEC = 1,
    TIME_MSEC = 1'000,
    TIME_USEC = 1'000'000,
    TIME_NSEC = 1'000'000'000,
};

template <TimeUnit SRC, TimeUnit DEST>
double TimeConvert(double src) {
    if constexpr (DEST == SRC) {
        return src;
    } else if constexpr (DEST > SRC) {
        return src * int64_t(DEST / SRC);
    } else {
        return src / int64_t(SRC / DEST);
    }
}
} // namespace moke
