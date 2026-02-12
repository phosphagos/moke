#include "moke/native/timer.hpp"
#include "moke/runtime.hpp"

namespace moke {
device_timer::device_timer() {
    check_status(hipEventCreate(&m_begin));
    check_status(hipEventCreate(&m_end));
}

device_timer::~device_timer() {
    check_status(hipEventDestroy(m_begin));
    check_status(hipEventDestroy(m_end));
}

void device_timer::start() {
    check_status(hipEventRecord(m_begin));
}

void device_timer::stop() {
    check_status(hipEventRecord(m_end));
}

auto device_timer::elapsed() const -> duration_t {
    float elapsed_ms;
    check_status(hipDeviceSynchronize());
    check_status(hipEventElapsedTime(&elapsed_ms, m_begin, m_end));
    return duration_t(elapsed_ms);
}
} // namespace moke
