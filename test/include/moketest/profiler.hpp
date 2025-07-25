#pragma once
#include "moketest/result.hpp"
#include <cstdio>

namespace moke {
class HostProfiler {
public:
    class Instance;

private:
    Instance *instance;

public:
    HostProfiler();
    void start();
    void finalize();
    PerformanceResult get(size_t ops, size_t io_bytes, size_t loops = 1) const;
};

class DeviceProfiler {
public:
    class Instance;

private:
    Instance *instance;

public:
    DeviceProfiler();
    void start();
    void finalize();
    PerformanceResult get(size_t ops, size_t io_bytes, size_t loops = 1) const;
};
} // namespace moke
