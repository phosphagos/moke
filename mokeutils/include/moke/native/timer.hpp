#pragma once
#include "moke/runtime.hpp"
#include <chrono>
#include <ratio>

namespace moke {
template <class period = std::ratio<1>>
using duration = std::chrono::duration<double, period>;

template <class dest_period, class src_period>
duration<dest_period> duration_cast(duration<src_period> dur) {
    return std::chrono::duration_cast<duration<dest_period>>(dur);
}

namespace device {
#if defined MOKE_PLATFORM_CUDA
    using stream_t = cudaStream_t;
    using event_t = cudaEvent_t;
#elif defined MOKE_PLATFORM_HIP
    using stream_t = hipStream_t;
    using event_t = hipEvent_t;
#endif
} // namespace device

template <class concrete_timer>
struct timer {
    template <class Period, class Func>
    static auto invoke(Func &&func) {
        concrete_timer timer;
        timer.start();
        func();
        timer.stop();
        return duration_cast<Period>(timer.elapsed());
    }
};

class host_timer : public timer<host_timer> {
public:
    using clock_t = std::chrono::high_resolution_clock;
    using timestamp_t = std::chrono::time_point<clock_t>;
    using period_t = clock_t::period;
    using duration_t = moke::duration<period_t>;

private:
    clock_t m_clock{};
    timestamp_t m_begin{};
    timestamp_t m_end{};

public:
    host_timer() = default;

    ~host_timer() = default;

    void start() { m_begin = m_clock.now(); }

    void stop() { m_end = m_clock.now(); }

    duration_t elapsed() const { return duration_t(m_end - m_begin); }
};

class device_timer : public timer<device_timer> {
public:
    using period_t = std::milli;
    using duration_t = moke::duration<period_t>;

private:
    device::event_t m_begin{nullptr};
    device::event_t m_end{nullptr};

public:
    device_timer();

    ~device_timer();

    device_timer(const device_timer &) = delete;

    device_timer(device_timer &&) = delete;

    void start();

    void stop();

    duration_t elapsed() const;
};
} // namespace moke
