#pragma once
#include "moke/native.hpp"
#include "moke/utils/memory.hpp"
#include "moke/utils/vector.hpp"
#include <cmath>

namespace moke::compare {
template <class T, class MS>
class all_equals {
private:
    const T *m_baseline;
    size_t m_length;

public:
    all_equals(MS, const T *baseline, size_t length) noexcept
            : m_baseline{baseline}, m_length{length} {}

    template <class Container>
    all_equals(const Container &baseline) noexcept
            : all_equals{baseline.mem_space(), baseline.data(), baseline.size()} {}

    bool operator()(MS memory_space, const T *result, size_t length) const {
        if (length != m_length) { return false; }
        return memory_compare(memory_space, result, m_baseline, length);
    }

    template <class Container>
    bool operator()(const Container &result) const {
        return operator()(result.mem_space(), result.data(), result.size());
    }
};

template <template <class, class> class Container, class T, class MS>
all_equals(const Container<T, MS> &) -> all_equals<T, MS>;

template <class T, class MS, bool REL = true>
class all_close {
private:
    constexpr static C<REL> m_relative{};
    const T *m_baseline;
    size_t m_length;
    float m_epsilon;

public:
    all_close(MS, const T *baseline, size_t length, float epsilon = 1e-6, C<REL> = {}) noexcept
            : m_baseline{baseline}, m_length{length}, m_epsilon(epsilon) {}

    template <class Container>
    all_close(const Container &baseline, float epsilon = 1e-6, C<REL> = {}) noexcept
            : all_close{baseline.mem_space(), baseline.data(), baseline.size(), epsilon} {}

    bool operator()(MS memory_space, const T *result, size_t length) const {
        if (length != m_length) { return false; }
        return compare_all_close(memory_space, result, m_baseline, length, m_relative, m_epsilon);
    }

    template <class Container>
    bool operator()(const Container &result) const {
        return operator()(result.mem_space(), result.data(), result.size());
    }
};

template <template <class, class> class Container, class T, class MS>
all_close(const Container<T, MS> &, float = 1e-6) -> all_close<T, MS, true>;

template <template <class, class> class Container, class T, class MS>
all_close(const Container<T, MS> &, float, C<true>) -> all_close<T, MS, true>;

template <template <class, class> class Container, class T, class MS>
all_close(const Container<T, MS> &, float, C<false>) -> all_close<T, MS, false>;
} // namespace moke::compare
