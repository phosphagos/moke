#pragma once
#if defined MOKE_PLATFORM_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace moke {
using half_t = __half;
using bfloat16_t = __nv_bfloat16;
} // namespace moke
#elif defined MOKE_PLATFORM_HIP
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

namespace moke {
using half_t = __half;
using bfloat16_t = __hip_bfloat16;
} // namespace moke
#endif // MOKE_PLATFORM
