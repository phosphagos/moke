#include "moke/common/runtime.hpp"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace moke {
template <> void CheckStatus(cudaError_t status, std::source_location loc) {
    if (status == cudaSuccess) { return; }
    std::printf("cuda error occurred: %s\n", cudaGetErrorString(status));
    std::printf("    in function \"%s\"\n", loc.function_name());
    std::printf("    at %s:%d\n", loc.file_name(), loc.line());
    std::printf("\n");
    std::printf("exited with error code %d\n", (int)status);
    std::exit((int)status);
}
} // namespace moke
