#pragma once
#include "moke/common.hpp"

#if defined MOKE_PLATFORM_CUDA
#include <cuda_runtime.h>
#elif defined MOKE_PLATFORM_HIP
#include <hip/hip_runtime.h>
#endif

namespace moke {
template <class Status>
void check_status(Status status);

void sync_device();
} // namespace moke
