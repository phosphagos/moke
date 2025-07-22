#pragma once
#include "moke/common/def.hpp"
#include "moketest/result.hpp"
#include <cstdio>
#include <cstdlib>

namespace moke {
class AbsoluteErrorComparator {
private:
    double m_threshold;

public:
    AbsoluteErrorComparator(double error = 1e-6) noexcept : m_threshold{error} {}

    template <class T>
    AccuracyResult operator()(const T *result, const T *baseline, size_t length) {
        double max_error = 0;
        size_t index = 0;
        for (size_t i = 0; i < length; i++) {
            double curr_error = std::abs(result[i] - baseline[i]);
            if (curr_error > max_error) {
                max_error = curr_error;
                index = i;
            }
        }

        return {m_threshold, max_error, index};
    }
};

class RelativeErrorComparator {
private:
    double m_threshold;

public:
    template <class T>
    struct Result {
    };

    RelativeErrorComparator(double threshold = 1e-6) noexcept : m_threshold{threshold} {}

    template <class T>
    AccuracyResult operator()(const T *result, const T *baseline, size_t length) {
        double max_error = 0;
        size_t index = 0;
        for (size_t i = 0; i < length; i++) {
            double curr_error = std::abs((result[i] - baseline[i]) / (double)baseline[i]);
            if (curr_error > max_error) {
                max_error = curr_error;
                index = i;
            }
        }

        return {m_threshold, max_error, index};
    }
};
} // namespace moke
