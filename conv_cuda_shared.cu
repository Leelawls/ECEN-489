#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"

#define TILE_WIDTH 16
#define MAX_KERNEL_SIZE 7  // Support up to 7x7 kernel

__global__ void conv2d_shared(
    const float* input, const float* kernel, float* output,
    int height, int width, int kernel_size
) {
    int pad = kernel_size / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - pad;
    int col_i = col_o - pad;

    __shared__ float tile[TILE_WIDTH + MAX_KERNEL_SIZE - 1][TILE_WIDTH + MAX_KERNEL_SIZE - 1];

    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
        tile[ty][tx] = input[row_i * width + col_i];
    } else {
        tile[ty][tx] = 0.0f;
    }

    __syncthreads();

    float sum = 0.0f;

    if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < height && col_o < width) {
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                sum += tile[ty + i][tx + j] * kernel[i * kernel_size + j];
            }
        }
        output[row_o * width + col_o] = sum;
    }
}

int main() {
    int width, height, channels;
    unsigned char* img = stbi_load("face.jpg", &width, &height, &channels, 1);
    if (!img) {
        printf("Failed to load image\n");
        return -1;
    }

    int img_size = width * height;
    float *h_input = new float[img_size];
    float *h_output = new float[img_size];

    for (int i = 0; i < img_size; i++)
        h_input[i] = static_cast<float>(img[i]);

    int kernel_size = 3;
    float h_kernel[] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    float *d_input, *d_output, *d_kernel;
    cudaMalloc(&d_input, sizeof(float) * img_size);
    cudaMalloc(&d_output, sizeof(float) * img_size);
    cudaMalloc(&d_kernel, sizeof(float) * kernel_size * kernel_size);

    cudaMemcpy(d_input, h_input, sizeof(float) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kernel_size * kernel_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH + kernel_size - 1, TILE_WIDTH + kernel_size - 1);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    conv2d_shared<<<dimGrid, dimBlock>>>(d_input, d_kernel, d_output, height, width, kernel_size);
    cudaDeviceSynchronize();
    // --- CUDA Timing Start ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // -------------------------

    conv2d_shared<<<dimGrid, dimBlock>>>(d_input, d_kernel, d_output, height, width, kernel_size);
    cudaDeviceSynchronize();

    // --- CUDA Timing Stop ---
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA kernel execution time: %.3f ms\n", milliseconds);
    // ------------------------

    cudaMemcpy(h_output, d_output, sizeof(float) * img_size, cudaMemcpyDeviceToHost);

    unsigned char* result_img = new unsigned char[img_size];
    for (int i = 0; i < img_size; i++) {
        result_img[i] = (unsigned char)fminf(fmaxf(h_output[i], 0.0f), 255.0f);
    }

    stbi_write_png("output.png", width, height, 1, result_img, width);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    delete[] h_input;
    delete[] h_output;
    delete[] result_img;
    stbi_image_free(img);

    return 0;
}

