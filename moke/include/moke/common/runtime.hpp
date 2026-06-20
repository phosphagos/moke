#pragma once
#include "moke/common/macros.hpp"
#include <cstdio>
#include <source_location>
#include <stdexcept>
#include <string>

#if defined MOKE_PLATFORM_CUDA
#include <cuda_runtime.h>
#elif defined MOKE_PLATFORM_HIP
#include <hip/hip_runtime.h>
#endif

namespace moke {
/// @brief checks if the status is success
bool status_success(auto status);

/// @brief getting the error catagory of a status
const char *status_category(auto status);

/// @brief getting the error info of a status
std::string status_message(auto status);

inline void check_status(auto status, std::source_location location = std::source_location::current()) {
    if (status_success(status)) { return; }
    auto category = status_category(status);
    auto errinfo = status_message(status);
    std::fprintf(stderr, "%s: %s\n", category, errinfo.c_str());
    std::fprintf(stderr, "    at %s\n", location.file_name());
    std::fprintf(stderr, "    at %s:%d:%d\n", location.file_name(), location.line(), location.column());
    throw std::runtime_error(errinfo);
}

inline void sync_device() {
#if defined MOKE_PLATFORM_CUDA
    check_status(cudaDeviceSynchronize());
#elif defined MOKE_PLATFORM_HIP
    check_status(hipDeviceSynchronize());
#endif
}

#if defined MOKE_PLATFORM_CUDA
template <> inline bool status_success(cudaError_t status) { return status == cudaSuccess; }
template <> inline const char *status_category(cudaError_t status) { return "cuda_runtime_error"; }
template <> inline std::string status_message(cudaError_t status) { return cudaGetErrorString(status); }
#elif defined MOKE_PLATFORM_HIP
template <> inline bool status_success(hipError_t status) { return status == hipSuccess; }
template <> inline const char *status_category(hipError_t status) { return "hip_runtime_error"; }
template <> inline std::string status_message(hipError_t status) { return hipGetErrorString(status); }
#endif
} // namespace moke
