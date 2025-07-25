message("[MOKE] cuda backend enabled")

set(MOKE_PLATFORM cuda)
add_definitions(-DMOKE_PLATFORM_CUDA)
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY STATIC)

if (DEFINED MOKE_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES ${MOKE_ARCH})
    message("[MOKE] cuda architecture set to '${MOKE_ARCH}'")
endif()
