#pragma once
#include "moke/common/def.hpp"
#include <cstdio>

namespace moke {
struct AccuracyResult {
    double threshold;
    double max_error;
    size_t index;

    explicit operator bool() const { return max_error <= threshold; }

    void print(FILE *out = stdout) const;
};

struct PerformanceResult {
    double kernel_time;   // in us
    double compute_force; // in GFLOPS
    double bandwidth;     // in GB/s

    void print(FILE *out = stdout) const;
};
} // namespace moke
