#include "moke/common.hpp"
#include "moke/kernels.hpp"
#include <cstdint>

namespace moke {
__global__ void color_to_grayscale(const DeviceTensorView<uint8_t, 3> input, DeviceTensorView<uint8_t, 2> output) {
    auto [x, y, _] = blockIdx * blockDim + threadIdx;
    if (x >= input.shape(1) || y >= input.shape(0)) { return; }

    uint16_t r = input[y][x][0];
    uint16_t g = input[y][x][1];
    uint16_t b = input[y][x][2];
    output[y][x] = 0.21f * r + 0.71f * g + 0.07f * b;
}

template <>
void ColorToGrayscale(const DeviceTensorView<uint8_t, 3> &input, DeviceTensorView<uint8_t, 2> &output) {
    if (input.shape(0) != output.shape(0) || input.shape(1) != output.shape(1)) {
        std::printf("Error: size of input and output does not match.\n");
        return;
    } else if (input.shape(2) != 3) {
        std::printf("Error: only 3 channels of r,g,b in input is supported.\n");
        return;
    }

    dim3 total_dim = {uint32_t(input.shape(0)), uint32_t(input.shape(1)), 1};
    dim3 block_dim = {32, 32, 1};
    return color_to_grayscale<<<ceil_div(total_dim, block_dim), block_dim>>>(input, output);
}
} // namespace moke
