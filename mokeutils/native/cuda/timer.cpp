#include "moke/common.hpp"
#include "moke/native.hpp"

namespace moke {
device_timer::device_timer() {
    check_status(cudaEventCreate(&m_begin));
    check_status(cudaEventCreate(&m_end));
}

device_timer::~device_timer() {
    check_status(cudaEventDestroy(m_begin));
    check_status(cudaEventDestroy(m_end));
}

void device_timer::start() {
    check_status(cudaEventRecord(m_begin));
}

void device_timer::stop() {
    check_status(cudaEventRecord(m_end));
}

auto device_timer::elapsed() const -> duration_t {
    float elapsed_ms;
    check_status(cudaDeviceSynchronize());
    check_status(cudaEventElapsedTime(&elapsed_ms, m_begin, m_end));
    return duration_t(elapsed_ms);
}
} // namespace moke
