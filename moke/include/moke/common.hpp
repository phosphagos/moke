#pragma once
#include <cstddef>
#include <cstdint>

#if defined __CUDACC__ || defined __HIPCC__
#define MOKE_UNIFIED __host__ __device__
#define MOKE_DEVICE __device__
#define MOKE_HOST __host__
#define MOKE_KERNEL __global__
#else
#define MOKE_UNIFIED
#define MOKE_HOST
#define MOKE_DEVICE
#define MOKE_KERNEL
#endif

#define MOKE_INLINE MOKE_UNIFIED inline
#define MOKE_CONSTEXPR MOKE_INLINE constexpr
#define MOKE_CAPI extern "C"

namespace moke {
using size_t = std::size_t;
using index_t = std::int64_t;
} // namespace moke
