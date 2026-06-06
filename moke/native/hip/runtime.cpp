#include "moke/common.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>

namespace moke {
template <> void check_status(hipError_t status, std::source_location location) {
    if (status == hipSuccess) { return; }
    auto errinfo = hipGetErrorString(status);
    std::fprintf(stderr, "hip runtime error: %s\n", errinfo);
    std::fprintf(stderr, "    at %s\n", location.file_name());
    std::fprintf(stderr, "    at %s:%d:%d\n", location.file_name(), location.line(), location.column());
    throw std::runtime_error(errinfo);
}

void sync_device() {check_status(hipDeviceSynchronize()); }
} // namespace moke
