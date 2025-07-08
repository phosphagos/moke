message("hip backend enabled")
add_definitions(-DMOKE_PLATFORM_HIP)
add_definitions(-DMOKE_PLATFORM="hip")
find_package(hip REQUIRED)

enable_language(HIP)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_EXTENSIONS OFF)

if (DEFINED MOKE_ARCH)
    set(CMAKE_HIP_ARCHITECTURES ${MOKE_ARCH})
    message("hip architecture set to ${MOKE_ARCH}")
endif()
