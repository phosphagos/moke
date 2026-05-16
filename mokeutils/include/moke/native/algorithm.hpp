#pragma once
#include "moke/common.hpp"
#include "moke/meta.hpp"
#include "moke/native/memory.hpp"
#include <cmath>
#include <cstdlib>
#include <random>

namespace moke {
template <class T> MOKE_INLINE bool absolute_close(T a, T b, float epsilon) {
    return a == b ? true : fabsf(a - b) <= epsilon;
}

template <class T> MOKE_INLINE bool relative_close(T a, T b, float epsilon) {
    return a == b ? true : fabsf(a - b) <= epsilon * fmax(fabsf(a), fabsf(b));
}

template <> MOKE_INLINE bool absolute_close(double a, double b, float epsilon) {
    return a == b ? true : fabs(a - b) <= epsilon;
}

template <> MOKE_INLINE bool relative_close(double a, double b, float epsilon) {
    return a == b ? true : fabs(a - b) <= epsilon * fmax(fabs(a), fabs(b));
}

template <class T, bool REL = true>
bool compare_all_close(host_memory_t, const T *lhs, const T *rhs, size_t length, C<REL> = {}, float epsilon = 1e-6) {
    auto compare_func = REL ? relative_close<T> : absolute_close<T>;
    for (int i = 0; i < length; i++) {
        if (!compare_func(lhs[i], rhs[i], epsilon)) { return false; }
    }
    return true;
}

template <class T, bool REL = true>
bool compare_all_close(device_memory_t, const T *lhs, const T *rhs, size_t length, C<REL> = {}, float epsilon = 1e-6);

template <class T>
void fill_random(host_memory_t, T *dest, size_t length, float min, float max, uint32_t seed) {
    std::mt19937 gen{seed};
    std::uniform_real_distribution<> dist{min, max};
    for (int i = 0; i < length; i++) { dest[i] = dist(gen); }
}

template <class T>
void fill_random(host_memory_t, T *dest, size_t length, float min, float max, uint32_t seed, uint8_t bits) {
    std::mt19937 gen{seed};
    std::uniform_real_distribution<> dist{min, max};
    for (int i = 0; i < length; i++) {
        float rnd = dist(gen);
        rnd = std::round(rnd * (1 << bits)) / (1 << bits);
        dest[i] = (T)rnd;
    }
}

template <class T>
void fill_random(device_memory_t, T *dest, size_t length, float min, float max, uint32_t seed);

template <class T>
void fill_random(device_memory_t, T *dest, size_t length, float min, float max, uint32_t seed, uint8_t bits);
} // namespace moke
