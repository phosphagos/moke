find_package(hip REQUIRED)

enable_language(HIP)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

foreach(MOKE_HIP_INCLUDE_DIR IN LISTS hip_INCLUDE_DIRS)
    # for clangd to detect hip includes
    string(APPEND CMAKE_HIP_FLAGS " -isystem ${MOKE_HIP_INCLUDE_DIR}")
endforeach()
