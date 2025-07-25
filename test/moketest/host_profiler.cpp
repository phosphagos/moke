#include "moketest/common.hpp"
#include "moketest/common/time_unit.hpp"
#include "moketest/profiler.hpp"
#include <ctime>

namespace moke {
struct HostProfiler::Instance {
    clockid_t timer_id;
    struct timespec begin;
    struct timespec end;

    Instance() : timer_id{CLOCK_REALTIME} {}
};

HostProfiler::Instance *GetHostProfilerInstance() {
    static HostProfiler::Instance instance{};
    return &instance;
}

HostProfiler::HostProfiler() : instance{GetHostProfilerInstance()} {}

void HostProfiler::start() {
    clock_gettime(instance->timer_id, &instance->begin);
}

void HostProfiler::finalize() {
    clock_gettime(instance->timer_id, &instance->end);
}

PerformanceResult HostProfiler::get(size_t ops, size_t io_bytes, size_t loops) const {
    auto sec = instance->end.tv_sec - instance->begin.tv_sec;
    auto nsec = instance->end.tv_nsec - instance->begin.tv_nsec;
    double time_us = sec * 1e6 + nsec * 1e-3;
    if (loops != 1) { time_us /= loops; }

    return PerformanceResult{
            .kernel_time = time_us,
            .compute_force = ops / time_us * 1e-3,
            .bandwidth = io_bytes / time_us * 1e-3
    };
}

void PerformanceResult::print(FILE *out) const {
    std::fprintf(out, "      kernel_time: %.3lfus\n", kernel_time);
    std::fprintf(out, "    compute_force: %.3lfGFLOPS\n", compute_force);
    std::fprintf(out, "        bandwidth: %.3lfGB/s\n", bandwidth);
}
} // namespace moke
