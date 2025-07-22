#pragma once
#include "moke/tensor.hpp"
#include <cstdint>
#include <cstdlib>
#include <utility>

namespace moke {
class HostRandomGenerator {
private:
    uint32_t m_seed;

public:
    HostRandomGenerator() noexcept : m_seed{0} {}

    explicit HostRandomGenerator(uint32_t seed) noexcept : m_seed{seed} {}

#if __cplusplus >= 202002L  // for c++20
    // filling a host array.
    void fill(auto *data, size_t length, auto... args) {
        for (size_t i = 0; i < length; i++) {
            data[i] = one(args...);
        }
    }

    // filling a host tensor.
    void fill(const auto &data, auto... args) {
        return fill(data.data(), data.size(), args...);
    }

    // generating a host tensor
    template <class T, class... Shape, size_t NDims = sizeof...(Shape)>
    HostTensor<T, NDims> make_tensor(T min, T max, Shape &&...shape) {
        HostTensor<T, NDims> tensor{shape...};
        this->fill(tensor, min, max);
        return std::move(tensor);
    }
#endif

private:
    double one() { return rand_r(&m_seed) / double(RAND_MAX); }

    double one(double max) { return one() * max; }

    double one(double min, double max) { return min + one(max - min); }
};
} // namespace moke
