#pragma once
#include "moke/common/runtime/status.hpp"
#include "moke/common/runtime/hardware_info.hpp"

namespace moke {
inline void sync_device() {
#if defined MOKE_PLATFORM_CUDA
    check_status(cudaDeviceSynchronize());
#elif defined MOKE_PLATFORM_HIP
    check_status(hipDeviceSynchronize());
#endif
}
} // namespace moke
