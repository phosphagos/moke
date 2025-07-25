#pragma once
#include <cstdio>
#include <cstdlib>
#include <moke/kernels.hpp>
#include <moketest/executor.hpp>

using namespace moke;
class VectorAddExecutor : public Executor<VectorAddExecutor> {
private:
    size_t length;
    HostArray<float> hbuf_x, hbuf_y, hbuf_out, base_out;
    DeviceArray<float> dbuf_x, dbuf_y, dbuf_out;

public:
    VectorAddExecutor(size_t length, int warm_loop = 100, int perf_loop = 300)
            : Executor<VectorAddExecutor>{warm_loop, perf_loop}, length{length}, hbuf_x{length}, hbuf_y{length}, hbuf_out{length}, base_out{length}, dbuf_x{length}, dbuf_y{length}, dbuf_out{length} {}

    void init_input();

    void load();

    void store();

    void compute();

    void compute_baseline();

    bool compare_accuracy();

    size_t theoretical_ops();

    size_t theoretical_iobytes();
};
