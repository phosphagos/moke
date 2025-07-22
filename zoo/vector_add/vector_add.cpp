#include "vector_add.hpp"


void VectorAddExecutor::compute_baseline() {
    for (size_t i = 0; i < length; i++) {
        base_out[i] = hbuf_x[i] + hbuf_y[i];
    }
}

void VectorAddExecutor::compute() {
    VectorAdd<float>(dbuf_x, dbuf_y, dbuf_out);
}

void VectorAddExecutor::init_input() {
    HostRandomGenerator generator;
    generator.fill(hbuf_x);
    generator.fill(hbuf_y);
}

void VectorAddExecutor::load() {
    dbuf_x.load(hbuf_x);
    dbuf_y.load(hbuf_y);
}

void VectorAddExecutor::store() {
    dbuf_out.store(hbuf_out);
}

bool VectorAddExecutor::compare_accuracy() {
    RelativeErrorComparator comp{};
    auto result = comp(hbuf_out.data(), base_out.data(), length);
    result.print();
    return bool(result);
}

size_t VectorAddExecutor::theoretical_ops() {
    return length;
}

size_t VectorAddExecutor::theoretical_iobytes() {
    return 3 * length * sizeof(float);
}
