#pragma once
#include "moke/tensor.hpp"
#include "moketest/common.hpp"
#include "moketest/comparator.hpp"
#include "moketest/generator.hpp"
#include "moketest/profiler.hpp"

namespace moke {
template <class Base>
class Executor {
private:
    HostProfiler host_profiler{};
    DeviceProfiler device_profiler{};
    int warm_loop, perf_loop;

public:
    Executor(int perf_loop = 300, int warm_loop = 100)
            : warm_loop{warm_loop}, perf_loop{perf_loop} {}

    void init_input() { return self().init_input(); }

    void load() { return self().load(); }

    void store() { return self().store(); }

    void compute() { return self().compute(); }

    void compute_baseline() { return self().compute_baseline(); }

    bool compare_accuracy() { return self().compare_accuracy(); }

    size_t theoretical_ops() { return self().theoretical_ops(); }

    size_t theoretical_iobytes() { return self().theoretical_iobytes(); }

    void execute() {
        // init input data
        size_t ops = theoretical_ops();
        size_t iobytes = theoretical_iobytes();
        init_input();

        // get baseline output and performance
        host_profiler.start();
        compute_baseline();
        host_profiler.finalize();
        auto host_perf = host_profiler.get(ops, iobytes);

        // accuracy test
        load();
        compute();
        store();

        // accuracy output
        compare_accuracy();

        // performance warm_up
        for (int i = 0; i < warm_loop; ++i) {
            compute();
        }

        // performance profiling
        device_profiler.start();
        for (int i = 0; i < perf_loop; ++i) {
            compute();
        }
        device_profiler.finalize();
        auto device_perf = device_profiler.get(ops, iobytes, perf_loop);
        bool perf_pass = device_perf.kernel_time < host_perf.kernel_time;

        std::printf("Device Performance:\n");
        device_perf.print();
        std::printf("Host Performance:\n");
        host_perf.print();
        std::printf("Performance Acceleration: %s\n", perf_pass ? "PASS" : "FAILED");
        std::printf("    ratio: %.3lfx\n", host_perf.kernel_time / device_perf.kernel_time);
    }

private:
    Base &self() { return *static_cast<Base *>(this); }
};
} // namespace moke
