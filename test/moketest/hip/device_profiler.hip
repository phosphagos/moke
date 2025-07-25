#include "moke/common.hpp"
#include "moketest/common.hpp"
#include "moketest/profiler.hpp"

namespace moke {
struct DeviceProfiler::Instance {
    hipEvent_t begin;
    hipEvent_t end;

    Instance() {
        CheckStatus(hipEventCreate(&begin));
        CheckStatus(hipEventCreate(&end));
    }

    ~Instance() {
        CheckStatus(hipEventDestroy(end));
        CheckStatus(hipEventDestroy(begin));
    }
};

DeviceProfiler::Instance *GetDeviceProfilerInstance() {
    static DeviceProfiler::Instance instance{};
    return &instance;
}

DeviceProfiler::DeviceProfiler() : instance{GetDeviceProfilerInstance()} {}

void DeviceProfiler::start() {
    CheckStatus(hipStreamSynchronize(nullptr));
    CheckStatus(hipEventRecord(instance->begin));
}

void DeviceProfiler::finalize() {
    CheckStatus(hipEventRecord(instance->end));
    CheckStatus(hipStreamSynchronize(nullptr));
}

PerformanceResult DeviceProfiler::get(size_t ops, size_t io_bytes, size_t loops) const {
    float time_ms{0};
    CheckStatus(hipEventElapsedTime(&time_ms, instance->begin, instance->end));
    if (loops != 1) { time_ms /= loops; }

    return PerformanceResult{
            .kernel_time = time_ms * 1e3,
            .compute_force = ops / time_ms * 1e-6,
            .bandwidth = io_bytes / time_ms * 1e-6
    };
}
} // namespace moke
