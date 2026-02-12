#pragma once
#include "moke/native/timer.hpp"

namespace moke {
template <class Timer>
class profiler : Timer {
private:
    int m_warm_loops;
    int m_total_loops;

public:
    using time_metric_t = typename Timer::period_t;
    using perf_metric_t = std::ratio_divide<std::ratio<1>, time_metric_t>;

    profiler(int warmups = 100, int loops = 100) noexcept
            : Timer{}, m_warm_loops{warmups}, m_total_loops{warmups + loops} {}

    ~profiler() = default;

    class iterator;

    iterator begin() { return iterator{*this, 0}; }

    iterator end() { return iterator{*this, m_total_loops}; }

    profiler &loops(int warmups, int loops) {
        m_warm_loops = warmups;
        m_total_loops = warmups + loops;
        return *this;
    }

    template <class metric_t>
    double elapsed(metric_t) const {
        auto result = duration_cast<metric_t>(Timer::elapsed() / (m_total_loops - m_warm_loops));
        return result.count();
    }

    template <class dest_metric_t>
    double perf(size_t size, dest_metric_t metric) const {
        using time_metric_t = std::ratio_divide<std::ratio<1>, dest_metric_t>;
        return size / elapsed(time_metric_t{});
    }
};

template <class Timer>
class profiler<Timer>::iterator {
private:
    profiler &m_profiler;
    int loop_idx;

public:
    iterator(profiler &prof, int idx) noexcept : m_profiler{prof}, loop_idx{idx} {
        if (m_profiler.m_warm_loops == 0) { m_profiler.start(); }
    }

    iterator &operator++() {
        loop_idx++;

        if (loop_idx == m_profiler.m_warm_loops) {
            m_profiler.start();
        } else if (loop_idx == m_profiler.m_total_loops) {
            m_profiler.stop();
        }

        return *this;
    }

    bool operator==(const iterator &rhs) const {
        return loop_idx == rhs.loop_idx;
    }

    int operator*() const { return loop_idx; }
};

template class profiler<host_timer>;
using host_profiler = profiler<host_timer>;

template class profiler<device_timer>;
using device_profiler = profiler<device_timer>;
} // namespace moke
