find_package(CUDAToolkit REQUIRED)
add_library(moke "")
add_subdirectory(common)
add_subdirectory(kernels)

target_include_directories(moke PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(moke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(moke PUBLIC -DMOKE_PLATFORM_CUDA)
target_link_libraries(moke PUBLIC CUDA::cudart)
