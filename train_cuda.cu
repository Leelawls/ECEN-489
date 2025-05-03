#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"

#define KERNEL_SIZE 3
#define BLOCK_SIZE 16
#define NUM_IMAGES 128  // Simulate training-like workload

__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];

// Shared memory convolution kernel
__global__ void conv2d_shared(float* input, float* output, int width, int height) {
    __shared__ float tile[BLOCK_SIZE + KERNEL_SIZE - 1][BLOCK_SIZE + KERNEL_SIZE - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;
    int halo = KERNEL_SIZE / 2;

    // Load tile into shared memory
    if (x < width && y < height) {
        tile[ty][tx] = input[y * width + x];
    } else {
        tile[ty][tx] = 0.0f;
    }

    __syncthreads();

    float sum = 0.0f;
    if (tx >= halo && tx < BLOCK_SIZE + halo &&
        ty >= halo && ty < BLOCK_SIZE + halo &&
        x >= halo && x < width - halo &&
        y >= halo && y < height - halo) {

        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                sum += tile[ty - halo + i][tx - halo + j] * d_kernel[i * KERNEL_SIZE + j];
            }
        }

        output[y * width + x] = sum;
    }
}

int main() {
    int width, height, channels;

    // Load grayscale image
    unsigned char* img = stbi_load("face.jpg", &width, &height, &channels, 1);
    if (!img) {
        printf("Failed to load 'face.jpg'\n");
        return -1;
    }
    printf("Loaded image: %d x %d\n", width, height);
    int img_size = width * height;

    // Allocate and normalize input
    float* h_input = (float*)malloc(img_size * sizeof(float));
    float* h_output = (float*)malloc(img_size * sizeof(float));
    for (int i = 0; i < img_size; i++) {
        h_input[i] = img[i] / 255.0f;
    }

    // 3x3 blur kernel
    float h_kernel[KERNEL_SIZE * KERNEL_SIZE] = {
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9
    };
    cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(float) * KERNEL_SIZE * KERNEL_SIZE);

    // Device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, img_size * sizeof(float));
    cudaMalloc(&d_output, img_size * sizeof(float));
    cudaMemcpy(d_input, h_input, img_size * sizeof(float), cudaMemcpyHostToDevice);

    // CUDA kernel config
    dim3 dimBlock(BLOCK_SIZE + KERNEL_SIZE - 1, BLOCK_SIZE + KERNEL_SIZE - 1);
    dim3 dimGrid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warmup (1 run to avoid first-launch delay)
    conv2d_shared<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // Timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Run convolution NUM_IMAGES times
    for (int i = 0; i < NUM_IMAGES; i++) {
        conv2d_shared<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);
    }

    // Timing end after full device sync
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    printf("Convolved 'face.jpg' %d times\n", NUM_IMAGES);
    printf("Total GPU time: %.2f ms\n", elapsed);
    printf("Average per image: %.5f ms\n", elapsed / NUM_IMAGES);

    // Write result image
    cudaMemcpy(h_output, d_output, img_size * sizeof(float), cudaMemcpyDeviceToHost);

    unsigned char* out_img = (unsigned char*)malloc(img_size);
    for (int i = 0; i < img_size; i++) {
        float val = h_output[i];
        out_img[i] = (unsigned char)(fminf(fmaxf(val, 0.0f), 1.0f) * 255.0f);
    }

    stbi_write_png("face_convolved.png", width, height, 1, out_img, width);

    // Cleanup
    stbi_image_free(img);
    free(h_input);
    free(h_output);
    free(out_img);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

