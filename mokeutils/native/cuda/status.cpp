#include "moke/common.hpp"
#include "moke/runtime.hpp"

#include <cuda.h>
#include <curand.h>
#include <format>
#include <iostream>
#include <stdexcept>

#if __cplusplus >= 202302L
#include <stacktrace>
#define print_stacktrace(out) \
    out << "stacktrace:\n";   \
    out << std::stacktrace::current() << "\n";
#else
#define print_stacktrace(out)
#endif

namespace moke {
template <> void check_status(CUresult status) {
    if (status == CUDA_SUCCESS) { return; }
    const char *errinfo{nullptr};
    if (cuGetErrorString(status, &errinfo) != CUDA_SUCCESS) { errinfo = "Invalid CUDA driver error"; }
    auto errmsg = std::format("cuda driver error: {}", errinfo);

    std::cerr << errmsg << "\n";
    print_stacktrace(std::cerr);
    throw std::runtime_error(std::move(errmsg));
}

template <> void check_status(curandStatus_t status) {
    if (status == CURAND_STATUS_SUCCESS) { return; }

    auto errmsg = std::format("curand error: error code {}", (int)status);
    std::cerr << errmsg << "\n";
    print_stacktrace(std::cerr);
    throw std::runtime_error(std::move(errmsg));
}
} // namespace moke
