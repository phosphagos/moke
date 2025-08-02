#pragma once
#include <source_location>
#if defined(MOKE_PLATFORM_CUDA)
#include <cuda_runtime.h>
#elif defined(MOKE_PLATFORM_HIP)
#include <hip/hip_runtime.h>
#endif

namespace moke {
template <class Status>
void CheckStatus(Status, std::source_location = std::source_location::current());
} // namespace moke
