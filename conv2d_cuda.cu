#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16  // Block size for parallel execution

// CUDA Kernel for 2D Convolution
__global__ void conv2d_kernel(
    const float* input, const float* kernel, float* output,
    int in_channels, int out_channels, int height, int width,
    int kernel_size, int stride, int padding) {

    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;

    int out_height = (height - kernel_size + 2 * padding) / stride + 1;
    int out_width = (width - kernel_size + 2 * padding) / stride + 1;

    if (bx < out_width && by < out_height) {
        for (int oc = 0; oc < out_channels; oc++) {
            float sum = 0.0;

            for (int ic = 0; ic < in_channels; ic++) {
                for (int i = 0; i < kernel_size; i++) {
                    for (int j = 0; j < kernel_size; j++) {
                        int row = by * stride + i - padding;
                        int col = bx * stride + j - padding;

                        if (row >= 0 && row < height && col >= 0 && col < width) {
                            int input_idx = ic * height * width + row * width + col;
                            int kernel_idx = oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + i * kernel_size + j;
                            sum += input[input_idx] * kernel[kernel_idx];
                        }
                    }
                }
            }

            int output_idx = oc * out_height * out_width + by * out_width + bx;
            output[output_idx] = sum;
        }
    }
}

// Wrapper function for PyTorch
void conv2d_cuda(
    torch::Tensor input, torch::Tensor kernel, torch::Tensor output,
    int in_channels, int out_channels, int height, int width,
    int kernel_size, int stride, int padding) {

    const int BLOCK_DIM = BLOCK_SIZE;

    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);

    conv2d_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(), kernel.data_ptr<float>(), output.data_ptr<float>(),
        in_channels, out_channels, height, width, kernel_size, stride, padding
    );

    cudaStreamSynchronize(0)
}

// PyTorch Extension Module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_cuda", &conv2d_cuda, "CUDA optimized 2D Convolution");
}
