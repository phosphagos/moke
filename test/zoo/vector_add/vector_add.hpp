#pragma once
#include <cstdio>
#include <cstdlib>
#include <moke/kernels.hpp>
#include <moketest/executor.hpp>

using namespace moke;
template <class T>
class VectorAddExecutor : public Executor<VectorAddExecutor<T>> {
private:
    size_t length;
    HostArray<T> hbuf_x, hbuf_y, hbuf_out, base_out;
    DeviceArray<T> dbuf_x, dbuf_y, dbuf_out;

public:
    VectorAddExecutor(size_t length, int warm_loop = 100, int perf_loop = 300)
            : Executor<VectorAddExecutor>{warm_loop, perf_loop}, length{length},
              hbuf_x{length}, hbuf_y{length}, hbuf_out{length}, base_out{length},
              dbuf_x{length}, dbuf_y{length}, dbuf_out{length} {}

    void init_input() {
        HostRandomGenerator generator;
        generator.fill(hbuf_x);
        generator.fill(hbuf_y);
    }

    void load() {
        dbuf_x.load(hbuf_x);
        dbuf_y.load(hbuf_y);
    }

    void store() {
        dbuf_out.store(hbuf_out);
    }

    void compute() {
        VectorAdd<T>(dbuf_x, dbuf_y, dbuf_out);
    }

    void compute_baseline() {
        for (size_t i = 0; i < length; i++) {
            base_out[i] = hbuf_x[i] + hbuf_y[i];
        }
    }

    bool compare_accuracy() {
        RelativeErrorComparator comp{};
        auto result = comp(hbuf_out.data(), base_out.data(), length);
        result.print();
        return bool(result);
    }

    size_t theoretical_ops() {
        return length;
    }

    size_t theoretical_iobytes() {
        return 3 * length * sizeof(T);
    }
};
