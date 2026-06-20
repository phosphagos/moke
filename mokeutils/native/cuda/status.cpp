#include "moke/common.hpp"
#include "moke/native.hpp"

namespace moke {

template <> std::string status_message(CUresult status) {
    const char *errinfo{nullptr};
    if (cuGetErrorString(status, &errinfo) != CUDA_SUCCESS) {
        return "Invalid CUDA driver error";
    }

    return errinfo;
}

template <> std::string status_message(curandStatus_t status) {
    switch (status) {
        case CURAND_STATUS_SUCCESS:
            return "Success";
        case CURAND_STATUS_VERSION_MISMATCH:
            return "Version Mismatch";
        case CURAND_STATUS_NOT_INITIALIZED:
            return "Not Initialized";
        case CURAND_STATUS_ALLOCATION_FAILED:
            return "Allocation Failed";
        case CURAND_STATUS_TYPE_ERROR:
            return "Type Error";
        case CURAND_STATUS_OUT_OF_RANGE:
            return "Out Of Range";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "Length Not Mutliple";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "Double Precision Required";
        case CURAND_STATUS_LAUNCH_FAILURE:
            return "Launch Failure";
        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "Pre-existing Failure";
        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "Initialization Failed";
        case CURAND_STATUS_ARCH_MISMATCH:
            return "Arch Mismatch";
        case CURAND_STATUS_INTERNAL_ERROR:
            return "Internal Error";
        default:
            return "Unknown CURAND error";
    }
}
} // namespace moke
