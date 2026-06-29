#pragma once
#include "moke/common/runtime.hpp"

#ifdef MOKE_PLATFORM_CUDA
#include <cuda.h>
#include <curand.h>
#endif

namespace moke {
#ifdef MOKE_PLATFORM_CUDA
template <> inline bool status_success(CUresult status) { return status == CUDA_SUCCESS; }
template <> inline const char *status_category(CUresult status) { return "cuda_driver_error"; }
template <> std::string status_message(CUresult status);

template <> inline bool status_success(curandStatus_t status) { return status == CURAND_STATUS_SUCCESS; }
template <> inline const char *status_category(curandStatus_t status) { return "curand_error"; }
template <> std::string status_message(curandStatus_t status);
#endif
} // namespace moke
