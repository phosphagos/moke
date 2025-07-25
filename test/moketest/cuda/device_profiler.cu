#include "moke/common.hpp"
#include "moketest/common.hpp"
#include "moketest/profiler.hpp"

namespace moke {
struct DeviceProfiler::Instance {
    cudaEvent_t begin;
    cudaEvent_t end;

    Instance() {
        CheckStatus(cudaEventCreate(&begin));
        CheckStatus(cudaEventCreate(&end));
    }

    ~Instance() {
        CheckStatus(cudaEventDestroy(end));
        CheckStatus(cudaEventDestroy(begin));
    }
};

DeviceProfiler::Instance *GetDeviceProfilerInstance() {
    static DeviceProfiler::Instance instance{};
    return &instance;
}

DeviceProfiler::DeviceProfiler() : instance{GetDeviceProfilerInstance()} {}

void DeviceProfiler::start() {
    CheckStatus(cudaStreamSynchronize(nullptr));
    CheckStatus(cudaEventRecord(instance->begin));
}

void DeviceProfiler::finalize() {
    CheckStatus(cudaEventRecord(instance->end));
    CheckStatus(cudaStreamSynchronize(nullptr));
}

PerformanceResult DeviceProfiler::get(size_t ops, size_t io_bytes, size_t loops) const {
    float time_ms{0};
    CheckStatus(cudaEventElapsedTime(&time_ms, instance->begin, instance->end));
    if (loops != 1) { time_ms /= loops; }

    return PerformanceResult{
            .kernel_time = time_ms * 1e3,
            .compute_force = ops / time_ms * 1e-6,
            .bandwidth = io_bytes / time_ms * 1e-6
    };
}
} // namespace moke
