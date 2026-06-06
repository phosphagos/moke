#include "moke/common.hpp"
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>

namespace moke {
template <> void check_status(cudaError_t status, std::source_location location) {
    if (status == cudaSuccess) { return; }
    auto errinfo = cudaGetErrorString(status);
    std::fprintf(stderr, "cuda runtime error: %s\n", errinfo);
    std::fprintf(stderr, "    at %s\n", location.file_name());
    std::fprintf(stderr, "    at %s:%d:%d\n", location.file_name(), location.line(), location.column());
    throw std::runtime_error(errinfo);
}

void sync_device() { check_status(cudaDeviceSynchronize()); }
} // namespace moke
