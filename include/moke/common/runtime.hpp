#pragma once
#include <source_location>
#if defined(MOKE_PLATFORM_CUDA)
#include <cuda_runtime.h>
#elif defined(MOKE_PLATFORM_HIP)
#include <hip/hip_runtime.h>
#endif

namespace moke {
#if defined(MOKE_PLATFORM_CUDA)
using NativeStream = cudaStream_t;
using NativeEvent = cudaEvent_t;
using NativeStatus = cudaError_t;
#elif defined(MOKE_PLATFORM_HIP)
using NativeStream = hipStream_t;
using NativeEvent = hipEvent_t;
using NativeStatus = hipError_t;
#endif

template <class Status>
void CheckStatus(Status status, std::source_location = std::source_location::current());
} // namespace moke
