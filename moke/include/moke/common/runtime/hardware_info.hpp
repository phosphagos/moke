#pragma once
#include "moke/common/runtime.hpp"
#include <algorithm>
#include <thread>

namespace moke {

namespace device {
#if defined MOKE_PLATFORM_CUDA
    using hardware_info_t = cudaDeviceProp;
#elif defined MOKE_PLATFORM_HIP
    using hardware_info_t = hipDeviceProp_t;
#else
    struct hardware_info_t{};
#endif
} // namespace device

class hardware_info {
private:
    int m_device_id;
    device::hardware_info_t m_props;

public:
    MOKE_HOST hardware_info() : hardware_info(0) {}

    MOKE_HOST hardware_info(int device_id) : m_device_id{device_id} {
#if defined(MOKE_PLATFORM_CUDA)
        check_status(cudaGetDeviceProperties(&m_props, device_id));
#elif defined(MOKE_PLATFORM_HIP)
        check_status(hipGetDeviceProperties(&m_props, device_id));
#endif
    }

    int device_id() const { return m_device_id; }

    int compute_units() const {
#if defined(MOKE_PLATFORM_CUDA) || defined(MOKE_PLATFORM_HIP)
        return m_props.multiProcessorCount;
#else
        return std::thread::hardware_concurrency();
#endif
    }

    int blocks_per_compute_units(int block_size = 1) const {
#if defined(MOKE_PLATFORM_CUDA) || defined(MOKE_PLATFORM_HIP)
        return std::min(
                m_props.maxBlocksPerMultiProcessor,
                m_props.maxThreadsPerMultiProcessor / block_size
        );
#else
        return 1;
#endif
    }

    int block_concurrency(int block_size) const {
        return compute_units() * blocks_per_compute_units(block_size);
    }
};
} // namespace moke
