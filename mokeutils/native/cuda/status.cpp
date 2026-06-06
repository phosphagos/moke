#include "moke/common.hpp"

#include <cuda.h>
#include <curand.h>
#include <iostream>
#include <source_location>
#include <stdexcept>

namespace moke {
template <> void check_status(CUresult status, std::source_location location) {
    if (status == CUDA_SUCCESS) { return; }
    const char *errinfo{nullptr};
    if (cuGetErrorString(status, &errinfo) != CUDA_SUCCESS) { errinfo = "Invalid CUDA driver error"; }

    std::fprintf(stderr, "cuda driver error: %s\n", errinfo);
    std::fprintf(stderr, "    at %s\n", location.file_name());
    std::fprintf(stderr, "    at %s:%d:%d\n", location.file_name(), location.line(), location.column());
    throw std::runtime_error(errinfo);
}

template <> void check_status(curandStatus_t status, std::source_location location) {
    if (status == CURAND_STATUS_SUCCESS) { return; }

    std::fprintf(stderr, "curand error: error code %d", (int)status);
    std::fprintf(stderr, "    at %s\n", location.file_name());
    std::fprintf(stderr, "    at %s:%d:%d\n", location.file_name(), location.line(), location.column());
    throw std::runtime_error("curand error");
}
} // namespace moke
