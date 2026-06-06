#pragma once
#include <cstddef>
#include <cstdint>

#if defined MOKE_PLATFORM_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#elif defined MOKE_PLATFORM_HIP
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#endif // MOKE_PLATFORM

namespace moke {
using size_t = std::size_t;
using index_t = std::int64_t;

#if defined MOKE_PLATFORM_CUDA
using half_t = __half;
using bfloat16_t = __nv_bfloat16;
#elif defined MOKE_PLATFORM_HIP
using half_t = __half;
using bfloat16_t = __hip_bfloat16;
#endif // MOKE_PLATFORM
} // namespace moke
