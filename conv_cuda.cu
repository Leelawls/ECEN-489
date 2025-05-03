#include <iostream>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


#define BLOCK_SIZE 16

__global__ void conv2d_kernel(
    const float* input, const float* kernel, float* output,
    int height, int width, int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // col
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row

    int pad = kernel_size / 2;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int in_y = y + i - pad;
            int in_x = x + j - pad;

            if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                sum += input[in_y * width + in_x] * kernel[i * kernel_size + j];
            }
        }
    }

    output[y * width + x] = sum;
}

int main() {
    int width, height, channels;
    unsigned char* input_img = stbi_load("face.jpg", &width, &height, &channels, 1); // force grayscale
    if (!input_img) {
        std::cerr << "Failed to load image.\n";
        return -1;
    }

    // Convert to float and normalize
    int img_size = width * height;
    float* h_input = new float[img_size];
    for (int i = 0; i < img_size; ++i)
        h_input[i] = input_img[i] / 255.0f;

    int kernel_size = 3;
    float h_kernel[] = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };

    float* d_input;
    float* d_kernel;
    float* d_output;
    float* h_output = new float[img_size];

    size_t bytes_img = img_size * sizeof(float);
    size_t bytes_kernel = kernel_size * kernel_size * sizeof(float);

    cudaMalloc(&d_input, bytes_img);
    cudaMalloc(&d_kernel, bytes_kernel);
    cudaMalloc(&d_output, bytes_img);

    cudaMemcpy(d_input, h_input, bytes_img, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, bytes_kernel, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, height, width, kernel_size);
    cudaDeviceSynchronize();

    // Launch kernel with timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, height, width, kernel_size);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Convolution kernel execution time: " << milliseconds << " ms" << std::endl;


    cudaMemcpy(h_output, d_output, bytes_img, cudaMemcpyDeviceToHost);

    // Convert float output to uint8 for saving
    unsigned char* out_img = new unsigned char[img_size];
    float min_val = h_output[0], max_val = h_output[0];
    for (int i = 1; i < img_size; ++i) {
        if (h_output[i] < min_val) min_val = h_output[i];
        if (h_output[i] > max_val) max_val = h_output[i];
    }

    for (int i = 0; i < img_size; ++i) {
        out_img[i] = static_cast<unsigned char>(255 * (h_output[i] - min_val) / (max_val - min_val));
    }

    stbi_write_png("filtered_output.png", width, height, 1, out_img, width);

    // Cleanup
    stbi_image_free(input_img);
    delete[] h_input;
    delete[] h_output;
    delete[] out_img;
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}

